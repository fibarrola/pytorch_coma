import os
import torch
import pandas as pd
import torch.nn.functional as F
from Logger import logger, timestamp
from config_parser import read_config
from model import Coma
import mesh_operations
from helpers import *
import json

__author__ = ['Priyanka Patel', 'Rodrigo Bonazzola']

def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay


def save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))


def load_pretrained_model(config, checkpoint="last"):
    '''
    This function returns a model associated to a configuration file.
    :param config: the configuration of the run.
    :param checkpoint: if "last", the last checkpoint will be selected. If "best", the best checkpoint in terms of validation loss will be selected. If an integer, the corresponding checkpoint will be selected.
    :return: the model object
    '''
    raise NotImplementedError


def save_training_info(loss_info, filename):
    """
    :param loss_info: list of lists, containing six elements per "row" (the different losses), and one row per epoch
    :return:
    """
    #TODO: use a Pandas DataFrame for this
    column_names = ["epoch", "reconstruction_loss_training", "KLD_loss_training", "loss_training", "reconstruction_loss_eval", "KLD_loss_eval", "loss_eval"]
    with open(filename, "w") as handle:
        handle.write("%s\n" % ",".join(column_names))
        for row in loss_info:
            handle.write("%s\n" % ( ",".join(["%d"] + ["%.5f"]*(len(row)-1)) % tuple(row) ) )


def log_loss_info(losses, nepochs):
    losses.insert(1, nepochs) # include total number of epochs to the logged message
    logger.info(
        'Epoch {:d}/{:d}:  L1 loss + KL loss = Total Loss\n'
        '\tTrain set:\t {:.5f} + {:.5f} = {:.5f},\n'
        '\tValidation set:\t {:.5f} + {:.5f} = {:.5f}'
        .format(*losses)
    )
    
    
def lossFun(heartFake,heartReal)
    LossH = F.l1_loss(out, data.reshape(-1, coma.filters[0]))


def main(config):

    #This function is too long!
    #TODO: Use the Experiment class to modularize the different steps

    #TODO: Print the commit hashes of all the repositories used (this, VTK, psbody)
    #TODO: Print the library versions
    #TODO: Print the hashes of the data files

    logger.info('Current Git commit hash for repository pytorch_coma: %s' % get_current_commit_hash())

    output_dir = config['output_dir']
    format_tokens = {"TIMESTAMP": timestamp}
    #TODO: Put the format_tokens item into the configuration file
    output_dir = output_dir.format(**format_tokens)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    import logging
    log_file = config.get("log_file", None)
    if log_file is None:
        log_file = "{}/log".format(output_dir)

    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('Initializing parameters')
    eval_flag = config['eval']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    batch_size = config['batch_size']
    is_this_a_test = config['test']

    device = get_device()

    logger.info('Loading template mesh from %s' % config['template_fname'])
    template_mesh = get_template_mesh(config) # Used to extract the connectivity between vertices

    #TODO: Put this into a separate function
    logger.info('Generating transform matrices')
    # - A: adjacency matrices
    # - D: downsampling matrices
    # - U: upsampling matrices
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    logger.info('Loading dataset')
    #TODO important: make the partition random.
    dataset = load_cardiac_dataset(config)

    logger.info("Using %s meshes for training and %s for validation." % (dataset.nTraining, dataset.nVal))
    #TODO: I don't like the way I'm partitioning the data
    train_loader = get_loader(dataset.vertices_train, dataset.train_ids, batch_size=batch_size, num_workers=workers_thread, shuffle=True)
    val_loader = get_loader(dataset.vertices_val, dataset.val_ids, batch_size=1, num_workers=workers_thread, shuffle=False)
    test_loader = get_loader(dataset.vertices_test, dataset.test_ids, batch_size=1, num_workers=workers_thread, shuffle=False)

    logger.info('Loading CoMA model')
    #TODO: print architecture of the network
    tabDims = [len(pData.contXinds),len(pData.catgXinds),len(pData.contCinds),len(pData.catgCinds)]
    coma = Coma(config, D_t, U_t, A_t, num_nodes, tabDims)

    #TODO: Use a dictionary to map the configuration parameters to this behaviour
    #TODO: print configuration of the optimizer in the logs
    if opt == 'adam':
        optimizer = torch.optim.Adam(coma.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(coma.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    # To continue from a previous run saved in checkpoint_file
    checkpoint_file = config['checkpoint_file']
    if checkpoint_file:
        logging.info("Resuming training from previous execution: %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        coma.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # To find if this is fixed in pytorch # <<-- What does this comment mean? Did I make it?
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        start_epoch = 1

    coma.to(device)

    #TODO: change path for parameter in config
    with open("{}/config.json".format(output_dir), 'w') as fp:
        config["run_id"] = timestamp
        json.dump(config, fp, indent=4)

    if eval_flag:
        val_loss = evaluate(coma, val_loader, device)
        logger.info('val loss', val_loss)
        return

    best_val_loss = float('inf')
    loss_history = []

    for epoch in range(start_epoch, total_epochs + 1):

        # logger.info("Training for epoch %s" % epoch)
        loss_t, recon_loss_t, kld_loss_t = train(coma, train_loader, optimizer, device)
        loss_ev, recon_loss_ev, kld_loss_ev = evaluate(coma, val_loader, device)
        loss_history.append([epoch, recon_loss_t, kld_loss_t, loss_t, recon_loss_ev, kld_loss_ev, loss_ev])
        # TODO: improve the logging here.
        log_loss_info(loss_history[-1], total_epochs)

        save_model(coma, optimizer, epoch, loss_t, loss_ev, checkpoint_dir)

        # To get the best run in the evaluation subset.
        if loss_ev < best_val_loss:
            best_val_loss = loss_ev
            best_epoch = epoch
            best_model = coma

        if opt == 'sgd':
            adjust_learning_rate(optimizer, lr_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    save_training_info(loss_history, "{}/training_losses.csv".format(output_dir))
    logging.info("Training finished after %s epochs." % total_epochs)
    logging.info("The best model yields validation loss = %.5f (at epoch %s)." % (best_val_loss, str(best_epoch)))

    #TODO: change
    logging.info("Saving best model state in {}/best_model.pkl".format(output_dir))
    torch.save(best_model.state_dict(), "{}/best_model.pkl".format(output_dir))

    #TODO: replace path for field in config
    logging.info("Saving last model state in {}/best_model.pkl".format(output_dir))
    torch.save(coma.state_dict(), "{}/last_model.pkl".format(output_dir))

    # Change this to use the Experiment class.
    coma.eval()
    z_file = "{}/latent_space.csv".format(output_dir)
    perf_file = "{}/performance.csv".format(output_dir)
    z_columns = ["z" + str(i) for i in range(coma.z)]
    z_df = pd.DataFrame(None, columns=z_columns)
    perf_df = pd.DataFrame(None, columns=["mse"])

    subsets = ["train", "validation", "test"]
    loader_list = [train_loader, val_loader] if is_this_a_test else [train_loader, val_loader, test_loader]
    all_ids = []
    all_subsets = []

    logging.info("Calculating latent representation (z) and reconstruction performance measures.")

    with torch.no_grad():
        for i, loader in enumerate(loader_list):
            subset = subsets[i]
            all_subsets += [subset] * len(loader.dataset)
            for batch, ids in loader:
                # TODO: We are duplicating code here. The best would be to change encoder according to is_variational
                batch = batch.to(device)
                if coma.is_variational:
                    mu, log_var = coma.encoder(x=batch)
                    z = mu
                else:
                    z = coma.encoder(x=batch)

                z_df = z_df.append(pd.DataFrame(data=z.cpu().detach().numpy(), columns=z_columns))
                all_ids.extend([str(int(x)) for x in ids.numpy()])

                x = coma.decoder(z)

                mse = ((x - batch) ** 2).mean(axis=1).mean(axis=1).cpu().detach().numpy()

                # batch = batch[torch.randperm(len(batch)), :, :]
                #TODO: it's important to report MSE using shuffled data, for reference.
                # mse_shuffled = ((x - batch) ** 2).mean(axis=1).mean(axis=1).cpu().detach().numpy()
                perf_df = perf_df.append(pd.DataFrame(data=mse, columns=["mse"]))

    perf_df['ID'] = all_ids
    perf_df = perf_df.set_index('ID')
    perf_df['subset'] = all_subsets

    z_df['ID'] = all_ids
    z_df = z_df.set_index('ID')
    z_df['subset'] = all_subsets

    z_df.to_csv(z_file)
    perf_df.to_csv(perf_file)

    logging.info("Execution finished")

    # This a temporary (i.e. permanent) workaround to indicate that the execution finished
    from subprocess import call
    import shlex
    call(shlex.split("touch output/%s/.finished" % timestamp))


def train(coma, dataloader, optimizer, device):
    coma.train()
    total_loss = 0
    total_kld_loss = 0
    total_recon_loss = 0
    # for i, (data, ids) in enumerate(dataloader):
    # Let's say we've modified the dataloader to return
    # (ids,xContReal,xCatgReal,xMaskReal,data)
    for i, (ids,xContReal,xCatgReal,xMaskReal,data) in enumerate(dataloader):
        xContReal = xContReal.to(device)
        xCatgReal = xCatgReal.to(device)
        xMaskReal = xMaskReal.to(device)
        data = data.to(device)
        batch_size = xContReal.size(0)
        contCreal = Variable(xContReal[:,pData.contCinds])
        catgCreal = Variable(xCatgReal[:,pData.catgCinds])
        contXreal = Variable(xContReal[:,pData.contXinds])
        catgXreal = Variable(xCatgReal[:,pData.catgXinds])
        
        optimizer.zero_grad()
        out, contXfake, catgXfake, contCfake, catgCfake = coma(data, contXreal, catgXreal, contCreal, catgCreal)
        recon_loss = F.l1_loss(out, data.reshape(-1, coma.filters[0]))
        
        ### tabular losses
        with torch.no_grad(): # To avoid computing errors where inputs are missing
            contXrealAux = torch.where(contXreal==-1.,contXfake,contXreal) 
        tabContLoss = 50*F.mse_loss(contXfake,contXrealAux, reduction ='sum')
        tabCatgLoss = F.binary_cross_entropy(catgXfake, catgXreal, weight = xMaskReal, reduction='sum')
        
        total_recon_loss += batch_size * recon_loss.item()
        loss = recon_loss
        ### add tabular loss
        loss += tabContLoss + tabCatgLoss
        
        if coma.is_variational:
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kld_loss = -0.5 * torch.mean(torch.mean(1 + coma.log_var - coma.mu ** 2 - coma.log_var.exp(), dim=1), dim=0)
            loss += coma.kld_weight * kld_loss
            total_kld_loss += batch_size * coma.kld_weight * kld_loss.item()
        total_loss += batch_size * loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader.dataset), \
           total_recon_loss / len(dataloader.dataset), \
           total_kld_loss / len(dataloader.dataset)


def evaluate(coma, dataloader, device):
    #TODO: condense train and evaluate into a single function with "training" and "evaluate" modes
    coma.eval()
    total_loss = 0
    total_kld_loss = 0
    total_recon_loss = 0
    for i, (data, ids) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.size(0)
        with torch.no_grad():
            if coma.is_variational:
                mu, log_var = coma.encoder(x=data)
                z = mu # coma.sampling(mu, log_var)
                # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
                kld_loss = -0.5 * torch.mean(torch.mean(1 + coma.log_var - coma.mu ** 2 - coma.log_var.exp(), dim=1), dim=0)
                loss = coma.kld_weight * kld_loss
                total_kld_loss += batch_size * coma.kld_weight * kld_loss.item()
            else:
                z = coma.encoder(x=data)
                loss = 0
            out = coma.decoder(z)
            recon_loss = F.l1_loss(out, data.reshape(-1, coma.filters[0]))
            total_recon_loss += batch_size * recon_loss.item()
            loss += recon_loss
            total_loss += batch_size * loss.item()

    return total_loss / len(dataloader.dataset), \
           total_recon_loss / len(dataloader.dataset), \
           total_kld_loss / len(dataloader.dataset)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')

    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-od', '--output_dir', default=None, help='path where to store output')
    parser.add_argument('-id', '--data_dir', default=None, help='path where to fetch input data from')
    parser.add_argument('--test', default=False, action="store_true", help='Set this flag if you just want to test whether the code executes properly.')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        logger.error('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    ################################################################################
    ### Load configuration
    if not os.path.exists(args.conf):
        logger.error('Config not found' + args.conf)
    config = read_config(args.conf)

    if args.data_dir:
        config['data_dir'] = args.data_dir

    if args.output_dir:
        config['output_dir'] = args.output_dir

    if args.test:
        # some small values so that the execution ends quickly
        config['comments'] = "this is a test"
        config['nTraining'] = 32
        config['nVal'] = 80
        config['epoch'] = 3
    config['test'] = args.test

    main(config)