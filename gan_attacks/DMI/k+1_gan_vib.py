import os, sys
import time
import utils
import torch
import dataloader
import torchvision
from utils import *
from torch.autograd import grad
import torch.nn.functional as F
from discri import *
from generator import *
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError

sys.path.append('../BiDO')
import model


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]

    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise ArgumentTypeError('Boolean value expected.')
        
    parser = ArgumentParser(description='Step1: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist')
    parser.add_argument('--defense', default='reg', help='reg | vib')
    parser.add_argument('--root_path', default="./improvedGAN")
    parser.add_argument('--model_path', default='../BiDO/target_model')
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--acc', default=0, type=float)
    parser.add_argument('--sameR', type=str, help="'s' for same R and 'd' for different R", default = 's') 
    parser.add_argument('--prot', type=str, help="Protection mechanism, mrp or fe, default mrp", default = 'mrp')
    args = parser.parse_args()

    file = "./config/" + args.dataset + ".json"
    loaded_args = load_json(json_file=file)


    ############################# mkdirs ##############################
    #* Make specific dir to save results
    save_res_dir = os.path.join(args.root_path, f'{args.prot}_{args.sameR}')
    os.makedirs(save_res_dir, exist_ok=True)

    save_model_dir = os.path.join(args.root_path, args.dataset, args.defense)
    save_model_dir = os.path.join(save_model_dir, f'{args.prot}_{args.sameR}')
    os.makedirs(save_model_dir, exist_ok=True)
    # save_img_dir = "./improvedGAN/imgs_improved_{}".format(args.dataset)
    save_img_dir = os.path.join(save_res_dir, "imgs_improved_{}".format(args.dataset))
    os.makedirs(save_img_dir, exist_ok=True)

    #* Make log file
    log_file = os.path.join(save_model_dir, f"train_gan_{args.prot}_{args.sameR}.txt")
    utils.Tee(log_file, 'a+')
    ############################# mkdirs ##############################

    #* Modify image file path
    loaded_args["dataset"]["img_path"] = '../../datasets/CelebA/Img/img_align_celeba_png'
    file_path = loaded_args['dataset']['gan_file_path']
    stage = loaded_args['dataset']['stage']
    lr = loaded_args[stage]['lr']
    batch_size = loaded_args[stage]['batch_size']
    z_dim = loaded_args[stage]['z_dim']
    epochs = loaded_args[stage]['epochs']
    n_critic = loaded_args[stage]['n_critic']
    n_classes = loaded_args["dataset"]["n_classes"]

    model_name = loaded_args["dataset"]["model_name"]

    if args.dataset == 'celeba':
        if args.defense == 'vib':
            '''
             python k+1_gan_vib.py --defense=vib --beta=0.003 --ac=79.82 && 
             python k+1_gan_vib.py --defense=vib --beta=0.01  --ac=70.98 && 
             python k+1_gan_vib.py --defense=vib --beta=0.02  --ac=59.14 && 
             python recover_vib.py --defense=vib --beta=0.003 --ac=79.82  --iter=3000 --verbose && 
             python recover_vib.py --defense=vib --beta=0.01  --ac=70.98  --iter=3000 --verbose &&
             python recover_vib.py --defense=vib --beta=0.02  --ac=59.14  --iter=3000 --verbose
            '''
            # beta, ac = 3e-3, 79.82
            # beta, ac = 1e-2, 70.98
            # beta, ac = 2e-2, 59.14
            beta = args.beta
            ac = args.acc
            T = model.VGG16_vib(n_classes)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join("../BiDO/target_model/{}".format(args.dataset), args.defense,
                                  f"{model_name}_beta{beta:.3f}_{ac:.2f}.tar")

            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'])

            Gpath = os.path.join(save_model_dir, "{}_G_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)
            Dpath = os.path.join(save_model_dir, "{}_D_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)

        elif args.defense == 'reg':
            #* Modify target model
            num_classes = 1000
            if args.prot == 'mrp' and args.sameR == 's':
                ac = 86.30
                T = model.InceptionResnetV1MRP(pretrained="vggface2", sameR=True, mode="gan_prot", classify=True, num_classes=num_classes)
            elif args.prot == 'mrp' and args.sameR == 'd':
                ac = 96.97
                T = model.InceptionResnetV1MRP(pretrained="vggface2", sameR=False, mode="gan_prot", classify=True, num_classes=num_classes)
            elif args.prot == 'fe' and args.sameR == 's':
                ac = 3.56
                T = model.InceptionResnetV1FE(pretrained="vggface2", sameR=True, mode="gan_prot", classify=True, num_classes=num_classes)
            elif args.prot == 'fe' and args.sameR == 'd':
                ac = 100.00
                T = model.InceptionResnetV1FE(pretrained="vggface2", sameR=False, mode="gan_prot", classify=True, num_classes=num_classes)

            if args.prot == "unprot":
                ac = 87.50
                model_tar = f"{model_name}_{args.defense}_{ac:.2f}.tar"
                T = model.InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=num_classes)
            else:
                model_tar = f"{model_name}_{args.defense}_{ac:.2f}_{args.prot}_{args.sameR}.tar"

            print(f"Target model is {model_tar}")
            path_T = os.path.join(args.model_path, args.dataset, args.defense, model_tar)

            T = nn.DataParallel(T).cuda()
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'])

            Gpath = os.path.join(save_model_dir, "{}_G_reg_{:.2f}.tar").format(model_name, ac)
            Dpath = os.path.join(save_model_dir, "{}_D_reg_{:.2f}.tar").format(model_name, ac)

    elif args.dataset == 'mnist':
        if args.defense == 'vib':
            beta = args.beta = 0.1
            ac = args.acc = 99.06
            T = model.MCNN_vib(n_classes)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join("../BiDO/target_model/{}".format(args.dataset), args.defense,
                                  f"{model_name}_beta{beta:.3f}_{ac:.2f}.tar")

            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'])

            Gpath = os.path.join(save_model_dir, "{}_G_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)
            Dpath = os.path.join(save_model_dir, "{}_D_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)

        if args.defense == 'reg':
            ac = 99.94
            T = model.MCNN(n_classes)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join(args.model_path, args.dataset, args.defense, "MCNN_reg_99.94.tar")
            ckp_T = torch.load(path_T)
            check = T.load_state_dict(ckp_T['state_dict'])

            Gpath = os.path.join(save_model_dir, "{}_G_reg_{:.2f}.tar").format(model_name, ac)
            Dpath = os.path.join(save_model_dir, "{}_D_reg_{:.2f}.tar").format(model_name, ac)

    print("Saving to Gpath", Gpath)
    print("Saving to Dpath", Dpath)
    print("---------------------Training [%s]------------------------------" % stage)

    if args.dataset == 'celeba':
        G = Generator(z_dim)
        DG = MinibatchDiscriminator()

    elif args.dataset == 'mnist':
        G = GeneratorMNIST(z_dim)
        DG = MinibatchDiscriminator_MNIST()


    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()
    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    entropy = HLoss()

    #* Rename to dataloader_init
    #* Change mode from "gan" to "gan_prot"
    _, dataloader_init = utils.init_dataloader(loaded_args, file_path, batch_size, mode="gan_prot")

    step = 0
    for epoch in range(0, epochs):
        start = time.time()

        _, unlabel_loader1 = init_dataloader(loaded_args, file_path, batch_size, mode="gan_prot", iterator=True)
        _, unlabel_loader2 = init_dataloader(loaded_args, file_path, batch_size, mode="gan_prot", iterator=True)

        for i, (imgs, iden) in enumerate(dataloader_init):
            current_iter = epoch * len(dataloader_init) + i + 1
            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            #* Ignore identity label
            x_unlabel, _ = next(unlabel_loader1)
            x_unlabel2, _ = next(unlabel_loader2)

            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            #* Modify target model input/output format
            # y_prob = T(utils.low2high(imgs))
            y_prob = T(utils.low2high(imgs), iden)
            # y_prob = T(imgs)[-1]

            y = torch.argmax(y_prob, dim=1).view(-1)

            #* Downsize input imgs to 64 dim for discriminator
            _, output_label = DG(utils.low2high(imgs, dim=64))
            _, output_unlabel = DG(utils.low2high(x_unlabel, dim=64))
            _, output_fake = DG(f_imgs)

            loss_lab = softXEnt(output_label, y_prob)
            loss_unlab = 0.5 * (torch.mean(F.softplus(log_sum_exp(output_unlabel)))
                                - torch.mean(log_sum_exp(output_unlabel))
                                + torch.mean(F.softplus(log_sum_exp(output_fake))))
            dg_loss = loss_lab + loss_unlab

            acc = torch.mean((output_label.max(1)[1] == y).float())

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G
            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                mom_gen, output_fake = DG(f_imgs)
                #* Downsize input imgs to 64 dim for discriminator
                mom_unlabel, _ = DG(utils.low2high(x_unlabel2, dim=64))

                mom_gen = torch.mean(mom_gen, dim=0)
                mom_unlabel = torch.mean(mom_unlabel, dim=0)

                Hloss = entropy(output_fake)
                g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start

        print("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (epoch, interval, dg_loss, g_loss,
                                                                                   acc))

        if epoch + 1 >= 100:
            torch.save({'state_dict': G.state_dict()}, Gpath)
            torch.save({'state_dict': DG.state_dict()}, Dpath)

        if (epoch + 1) % 5 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(),
                               os.path.join(save_img_dir,
                                            f"improved_{args.dataset}_img_{args.defense}_{epoch + 1}.png"),
                               nrow=8)
