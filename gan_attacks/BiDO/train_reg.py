import torch, os, engine, model, utils, sys
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from util import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from copy import deepcopy

device = "cuda"
from csv_logger import CSVLogger, plot_csv
sys.path.append('../BiDO')
from model import InceptionResnetV1, InceptionResnetV1MRP, InceptionResnetV1FE
import time

def main(args, loaded_args, trainloader, testloader):
    n_classes = loaded_args["dataset"]["n_classes"]
    model_name = loaded_args["dataset"]["model_name"]
    weight_decay = loaded_args[model_name]["weight_decay"]
    momentum = loaded_args[model_name]["momentum"]
    n_epochs = loaded_args[model_name]["epochs"]
    lr = loaded_args[model_name]["lr"]
    milestones = loaded_args[model_name]["adjust_epochs"]

    if args.dataset == 'mnist':
        if model_name == "MCNN":
            net = model.MCNN(n_classes)
        elif model_name == "SCNN":
            net = model.SCNN(10)

    elif args.dataset == 'celeba':
        lr = 1e-2
        n_epochs = 50
        if model_name == "VGG16":
            net = model.VGG16(n_classes)

            optimizer = torch.optim.SGD(params=net.parameters(),
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay,
                                nesterov=True
                                )
        elif model_name == "FaceNet":
            #* Modify target model
            # net = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=n_classes)
            # T_path = os.path.join("target_model/celeba/reg/FaceNet_reg_87.50.tar")
            if args.prot == 'mrp' and args.sameR == 's':
                net = InceptionResnetV1MRP(pretrained="vggface2", sameR=True, classify=True, num_classes=n_classes)
                T_path = os.path.join("target_model/FaceNet_pytorch_classify_mrp_untrained.tar")
            elif args.prot == 'mrp' and args.sameR == 'd':
                net = InceptionResnetV1MRP(pretrained="vggface2", sameR=False, classify=True, num_classes=n_classes)
                T_path = os.path.join("target_model/FaceNet_pytorch_classify_mrp_diffR_untrained.tar")
            elif args.prot == 'fe' and args.sameR == 's':
                net = InceptionResnetV1FE(pretrained="vggface2", sameR=True, classify=True, num_classes=n_classes)
                T_path = os.path.join("target_model/FaceNet_pytorch_classify_fe_untrained.tar")
            elif args.prot == 'fe' and args.sameR == 'd':
                net = InceptionResnetV1FE(pretrained="vggface2", sameR=False, classify=True, num_classes=n_classes)
                T_path = os.path.join("target_model/FaceNet_pytorch_classify_fe_diffR_untrained.tar")
            elif args.prot == "unprot":
                net = InceptionResnetV1(pretrained="vggface2", sameR=False, classify=True, num_classes=n_classes)
                T_path = os.path.join("target_model/FaceNet_pytorch_classify_untrained.tar")
            print("Training classification model ", T_path)
            
            net = torch.nn.DataParallel(net).to(device)
            ckp_E = torch.load(T_path)
            net.load_state_dict(ckp_E['state_dict'], strict=False)
            for p in net.parameters():
                p.requires_grad_(False)
            logits_params = [p for name, p in net.named_parameters() if "logits" in name]
            for p in logits_params:
                p.requires_grad_(True)

            optimizer = torch.optim.SGD(params=logits_params,
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay,
                                nesterov=True
                                )
    

    scheduler = MultiStepLR(optimizer, milestones, gamma=0.2)

    criterion = nn.CrossEntropyLoss().cuda()
    # net = torch.nn.DataParallel(net).to(device)

    ################## viz ######################
    args.output_dir = os.path.join(args.model_dir, args.dataset, args.defense)
    os.makedirs(args.output_dir, exist_ok=True)
    #* Modify logger to log time, under new dir
    epoch_fieldnames = ['global_iteration', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'time']
    epoch_logger = CSVLogger(every=1,
                             fieldnames=epoch_fieldnames,
                             filename=os.path.join(model_path_subdir, f'epoch_log.csv'),
                             resume=0)
    print(f"Logging to {epoch_logger.filename}")
    ################## viz ######################
    best_acc = -1
    tf = time.time()
    for epoch in range(n_epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
        if args.sameR == 's':
            train_loss, train_acc = engine.train_reg(net, criterion, optimizer, trainloader)
            test_loss, test_acc = engine.test_reg(net, criterion, testloader)
        else:
            train_loss, train_acc = engine.train_reg_diffR(net, criterion, optimizer, trainloader)
            test_loss, test_acc = engine.test_reg_diffR(net, criterion, testloader)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = deepcopy(net)

        scheduler.step()

        ################################### viz ####################################
        if epoch % 1 == 0:
            interval = time.time() - tf
            epoch_logger.writerow({
                'global_iteration': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'time': interval,
            })
            plot_csv(epoch_logger.filename, os.path.join(args.output_dir, f'{args.prot}_{args.sameR}/', f'epoch_plots.jpeg'))
        ################################### viz ####################################

    print("best acc:", best_acc)
    utils.save_checkpoint({
        'state_dict': best_model.state_dict(),
    }, model_path, "{}_{}_{:.2f}_{}_{}.tar".format(model_name, args.defense, best_acc, args.prot, args.sameR))


    
if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentTypeError

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise ArgumentTypeError('Boolean value expected.')

    parser = ArgumentParser(description='train reg')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist | chestxray')
    parser.add_argument('--defense', default='reg', help='reg')
    parser.add_argument('--root_path', default='./', help='')
    parser.add_argument('--config_dir', default='./config', help='')
    parser.add_argument('--model_dir', default='./target_model', help='')
    parser.add_argument('--output_dir', default='./target_model/celeba/reg', help='')
    parser.add_argument('--sameR', type=str, help="'s' for same R and 'd' for different R", default = 's') 
    parser.add_argument('--prot', type=str, help="Protection mechanism, mrp or fe, default mrp", default = 'mrp')

    args = parser.parse_args()
    model_path = os.path.join(args.root_path, args.model_dir, args.dataset, args.defense)
    os.makedirs(model_path, exist_ok=True)
    #* Make specific subdir to save models
    model_path_subdir = os.path.join(model_path, f'{args.prot}_{args.sameR}')
    os.makedirs(model_path_subdir, exist_ok=True)
    
    file = os.path.join(args.config_dir, args.dataset + ".json")
    loaded_args = utils.load_json(json_file=file)
    #* Modify image file path
    loaded_args["dataset"]["img_path"] = '../../datasets/CelebA/Img/img_align_celeba_png'

    train_file = loaded_args['dataset']['train_file']
    test_file = loaded_args['dataset']['test_file']
    trainloader = utils.init_dataloader(loaded_args, train_file, mode="train")
    testloader = utils.init_dataloader(loaded_args, test_file, mode="test")

    main(args, loaded_args, trainloader, testloader)

