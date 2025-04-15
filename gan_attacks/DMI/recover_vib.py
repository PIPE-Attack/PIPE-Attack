import utils
from utils import *

from generator import *
from discri import *

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch, time, time, os, logging, statistics
import numpy as np
from generator import Generator
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError
import torch.nn.functional as F
from fid_score import calculate_fid_given_paths
# from fid_score_raw import calculate_fid_given_paths0

import sys

sys.path.append('../BiDO')
import model


def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps * std + mu


def dist_inversion(args, G, D, T, E, iden, lr=2e-2, lamda=100, iter_times=1500, clip_range=1,
                   improved=False, num_seeds=5, verbose=False):
    iden = iden.view(-1).long().to('cuda')
    criterion = nn.CrossEntropyLoss().to('cuda')
    bs = iden.shape[0]

    G.eval()
    D.eval()
    T.eval()
    E.eval()

    tf = time.time()

    # NOTE
    mu = Variable(torch.zeros(bs, 100), requires_grad=True)
    log_var = Variable(torch.zeros(bs, 100), requires_grad=True)

    params = [mu, log_var]
    solver = optim.Adam(params, lr=lr)

    for i in range(iter_times):
        z = reparameterize(mu, log_var)
        fake = G(z)
        if improved == True:
            _, label = D(fake)
        else:
            label = D(fake)

        #* Modify target model input/output format
        # out = T(utils.low2high(fake))
        if args.prot == "unprot":
            # out = T(fake)[-1]
            out = T(utils.low2high(fake))   
        else:
            out = T(utils.low2high(fake), iden) 

        for p in params:
            if p.grad is not None:
                p.grad.data.zero_()

        if improved:
            Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
        else:
            Prior_Loss = - label.mean()

        Iden_Loss = criterion(out, iden)

        Total_Loss = Prior_Loss + lamda * Iden_Loss

        Total_Loss.backward()
        solver.step()

        z = torch.clamp(z.detach(), -clip_range, clip_range).float()

        Prior_Loss_val = Prior_Loss.item()
        Iden_Loss_val = Iden_Loss.item()

        if (i + 1) % 500 == 0 and verbose:
            fake_img = G(z.detach())

            if args.dataset == 'celeba':
                #* Modify evaluator model output format
                eval_prob = E(utils.low2high(fake_img))
                # eval_prob = E(utils.low2high(fake_img))[-1]
            else:
                eval_prob = E(fake_img)[-1]

            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
            acc = iden.eq(eval_iden.long()).sum().item() * 100.0 / bs
            print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.4f}".format(i + 1, Prior_Loss_val,
                                                                                                Iden_Loss_val, acc))

    if verbose:
        interval = time.time() - tf
        print("Time:{:.2f}".format(interval))

    res = []
    res5 = []
    seed_acc = torch.zeros((bs, num_seeds))

    for random_seed in range(num_seeds):
        tf = time.time()
        z = reparameterize(mu, log_var)
        fake = G(z)

        if args.dataset == 'celeba':
            #* Modify evaluator model output format
            eval_prob = E(utils.low2high(fake))
            # eval_prob = E(utils.low2high(fake))[-1]
        else:
            eval_prob = E(fake)[-1]

        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)

        cnt, cnt5 = 0, 0
        for i in range(bs):
            gt = iden[i].item()
            sample = fake[i]
            save_tensor_images(sample.detach(),
                               os.path.join(args.save_img_dir, "attack_iden_{:03d}|{}.png".format(gt, random_seed + 1)))

            if eval_iden[i].item() == gt:
                seed_acc[i, random_seed] = 1
                cnt += 1
                best_img = G(z)[i]
                save_tensor_images(best_img.detach(), os.path.join(args.success_dir,
                                                                   "attack_iden_{:03d}|{}.png".format(gt,
                                                                                                      random_seed + 1)))
            _, top5_idx = torch.topk(eval_prob[i], 5)
            if gt in top5_idx:
                cnt5 += 1

        interval = time.time() - tf
        if verbose:
            print("Time:{:.2f}\tSeed:{}\tAcc:{:.4f}\t".format(interval, random_seed, cnt * 100.0 / bs))
        res.append(cnt * 100.0 / bs)
        res5.append(cnt5 * 100.0 / bs)

        torch.cuda.empty_cache()

    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
    acc_var = statistics.stdev(res)
    acc_var5 = statistics.stdev(res5)

    if verbose:
        print(f"Acc:{acc:.4f}\tAcc_5:{acc_5:.4f}\tAcc_var:{acc_var:.4f}\tAcc_var5:{acc_var5:.4f}")

    return acc, acc_5, acc_var, acc_var5


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
        
    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist')
    parser.add_argument('--defense', default='reg', help='reg | vib')
    parser.add_argument('--iter', default=5000, type=int)
    parser.add_argument('--improved_flag', action='store_true', default=True, help='use improved k+1 GAN')
    parser.add_argument('--root_path', default="./improvedGAN")
    parser.add_argument('--model_path', default='../BiDO/target_model')
    parser.add_argument('--save_img_dir', default='./attack_res/')
    parser.add_argument('--success_dir', default='')
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--acc', default=0, type=float)
    parser.add_argument('--verbose', action='store_true', help='') 
    parser.add_argument('--sameR', type=str, help="'s' for same R and 'd' for different R", default = 's') 
    parser.add_argument('--prot', type=str, help="Protection mechanism, mrp or fe, default mrp", default = 'mrp')

    args = parser.parse_args()

    z_dim = 100
    ############################# mkdirs ##############################
    log_path = os.path.join(args.root_path, "attack_logs")
    os.makedirs(log_path, exist_ok=True)
    #* Change log filename
    log_file = f"attack_{args.dataset}_{args.defense}_{args.prot}_{args.sameR}.txt"
    utils.Tee(os.path.join(log_path, log_file), 'a+')

    #* Make specific dir to save results
    save_model_dir = os.path.join(args.root_path, args.dataset, args.defense)
    save_model_dir = os.path.join(save_model_dir, f'{args.prot}_{args.sameR}')

    args.save_img_dir = os.path.join(args.save_img_dir, args.dataset, args.defense)
    save_res_dir = os.path.join(args.save_img_dir, f'{args.prot}_{args.sameR}')
    os.makedirs(save_res_dir, exist_ok=True)

    args.save_img_dir = save_res_dir
    args.success_dir = args.save_img_dir + "/res_success"
    os.makedirs(args.success_dir, exist_ok=True)

    args.save_img_dir = os.path.join(args.save_img_dir, 'all')
    os.makedirs(args.save_img_dir, exist_ok=True)

    ############################# mkdirs ##############################
    file = "./config/" + args.dataset + ".json"
    loaded_args = load_json(json_file=file)
    stage = loaded_args["dataset"]["stage"]
    model_name = loaded_args["dataset"]["model_name"]

    if args.dataset == 'celeba':
         #* Change evaluator model
        E = model.InceptionResnetV1()
        # E = model.FaceNet(1000)
        E = torch.nn.DataParallel(E).to('cuda')
        # path_E = './eval_ckp/FaceNet_95.88.tar'
        path_E = os.path.join("../BiDO/eval_ckp/FaceNet_pytorch.tar")
        ckp_E = torch.load(path_E)
        E.load_state_dict(ckp_E['state_dict'], strict=False)

        if args.defense == 'vib':
            # 0.003 78.70
            # 0.01 68.39
            # 0.02 53.94
            beta = args.beta
            ac = args.acc

            T = model.VGG16_vib(1000)
            T = torch.nn.DataParallel(T).to('cuda')
            path_T = os.path.join(args.model_path, args.dataset,
                                  args.defense, f"{model_name}_beta{beta:.3f}_{ac:.2f}.tar")
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            path_G = os.path.join(save_model_dir, "{}_G_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)
            path_D = os.path.join(save_model_dir, "{}_D_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)

        elif args.defense == 'reg':
            #* Change target model
            num_classes = 1000
            if args.prot == 'mrp' and args.sameR == 's':
                ac = 86.30
                T = model.InceptionResnetV1MRP(pretrained="vggface2", sameR=True, classify=True, num_classes=num_classes)
            elif args.prot == 'mrp' and args.sameR == 'd':
                ac = 96.97
                T = model.InceptionResnetV1MRP(pretrained="vggface2", sameR=False, classify=True, num_classes=num_classes)
            elif args.prot == 'fe' and args.sameR == 's':
                ac = 3.56
                T = model.InceptionResnetV1FE(pretrained="vggface2", sameR=True, classify=True, num_classes=num_classes)
            elif args.prot == 'fe' and args.sameR == 'd':
                ac = 100.00
                T = model.InceptionResnetV1FE(pretrained="vggface2", sameR=False, classify=True, num_classes=num_classes)
                
            if args.prot == "unprot":
                ac = 87.50
                model_tar = f"{model_name}_{args.defense}_{ac:.2f}.tar"
                T = model.InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=num_classes)
            else:
                model_tar = f"{model_name}_{args.defense}_{ac:.2f}_{args.prot}_{args.sameR}.tar"
                
            print(f"Target model is {model_tar}")
            path_T = os.path.join(args.model_path, args.dataset, args.defense, model_tar)
            
            T = torch.nn.DataParallel(T).to('cuda')
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            path_G = os.path.join(save_model_dir, "{}_G_reg_{:.2f}.tar").format(model_name, ac)
            print("Load G from:", path_G)
            path_D = os.path.join(save_model_dir, "{}_D_reg_{:.2f}.tar").format(model_name, ac)
            print("Load D from:", path_D)

        G = Generator(z_dim)
        G = torch.nn.DataParallel(G).to('cuda')
        D = MinibatchDiscriminator()
        D = torch.nn.DataParallel(D).to('cuda')

        ckp_G = torch.load(path_G)
        G.load_state_dict(ckp_G['state_dict'], strict=False)
        ckp_D = torch.load(path_D)
        D.load_state_dict(ckp_D['state_dict'], strict=False)

        ############         attack     ###########
        aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0

        # evaluate on the first 300 identities only
        #* Evaluate on all 1000 identities
        ids = 1000
        times = 20
        ids_per_time = ids // times
        iden = torch.from_numpy(np.arange(ids_per_time))
        for idx in range(times):
            if args.verbose:
                print("--------------------- Attack batch [%s]------------------------------" % idx)
                print(f"Attack on ids ids {idx*ids_per_time} to {(idx+1)*ids_per_time}")
                acc, acc5, var, var5 = dist_inversion(args, G, D, T, E, iden, lr=2e-2, lamda=100,
                                                    iter_times=args.iter, clip_range=1, improved=args.improved_flag,
                                                    num_seeds=5, verbose=args.verbose)

                iden = iden + ids_per_time
                aver_acc += acc / times
                aver_acc5 += acc5 / times
                aver_var += var / times
                aver_var5 += var5 / times
                res = aver_acc, aver_acc5, aver_var, aver_var5
                # np.save(os.path.join(args.save_img_dir, args.dataset, args.defense, f"res_batch{idx}.npy"), res)
                np.save(os.path.join(save_res_dir, f"res_batch{idx}.npy"), res)

        res = aver_acc, aver_acc5, aver_var, aver_var5
        np.save(os.path.join(save_res_dir, "res.npy"), res)
                
        #* Comment out FID computation
        # fid_value = calculate_fid_given_paths(args.dataset,
        #                                       [f'attack_res/{args.dataset}/trainset/',
        #                                        f'attack_res/{args.dataset}/{args.defense}/all/'],
        #                                       50, 1, 2048)
        # print(f'FID:{fid_value:.4f}')

        print("Average Acc:{:.2f}\tAverage Acc5:{:.2f}\tAverage Acc_var:{:.4f}\tAverage Acc_var5:{:.4f}".format(
            aver_acc,
            aver_acc5,
            aver_var,
            aver_var5))

    elif args.dataset == 'mnist':
        E = model.SCNN(10)
        path_E = './eval_ckp/SCNN_99.42.tar'
        ckp_E = torch.load(path_E)
        E = nn.DataParallel(E).to('cuda')
        E.load_state_dict(ckp_E['state_dict'])

        if args.defense == 'reg':
            ac = 99.94

            T = model.MCNN(5)
            T = torch.nn.DataParallel(T).to('cuda')
            path_T = os.path.join(args.model_path, args.dataset,
                                  args.defense, f"{model_name}_reg_{ac:.2f}.tar")
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            path_G = os.path.join(save_model_dir, "{}_G_reg_{:.2f}.tar").format(model_name, ac)
            path_D = os.path.join(save_model_dir, "{}_D_reg_{:.2f}.tar").format(model_name, ac)

        if args.defense == 'vib':
            beta = args.beta = 0.2
            ac = args.acc = 97.42

            T = model.MCNN_vib(5)
            T = torch.nn.DataParallel(T).to('cuda')
            path_T = os.path.join(args.model_path, args.dataset,
                                  args.defense, f"{model_name}_beta{beta:.3f}_{ac:.2f}.tar")
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            path_G = os.path.join(save_model_dir, "{}_G_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)
            path_D = os.path.join(save_model_dir, "{}_D_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)

        G = GeneratorMNIST(z_dim)
        G = torch.nn.DataParallel(G).to('cuda')
        D = MinibatchDiscriminator_MNIST()
        D = torch.nn.DataParallel(D).to('cuda')

        ckp_G = torch.load(path_G)
        G.load_state_dict(ckp_G['state_dict'], strict=False)
        ckp_D = torch.load(path_D)
        D.load_state_dict(ckp_D['state_dict'], strict=False)

        aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
        fid = []
        res_all = []

        K = 5
        for i in range(K):
            if args.verbose:
                print('-------------------------')
            iden = torch.from_numpy(np.arange(5))
            acc, acc5, var, var5 = dist_inversion(args, G, D, T, E, iden, lr=2e-2, lamda=100, iter_times=args.iter,
                                                  clip_range=1, improved=True, num_seeds=100, verbose=args.verbose)


            fid_value = calculate_fid_given_paths(args.dataset,
                                                  [f'attack_res/{args.dataset}/trainset/',
                                                   f'attack_res/{args.dataset}/{args.defense}/all/'],
                                                  50, 1, 2048)
            print(f'FID:{fid_value:.4f}')

            fid.append(fid_value)
            res_all.append([acc, acc5, var, var5])
            fid.append(fid_value)

        res = np.array(res_all).mean(0)
        avg_fid, var_fid = statistics.mean(fid), statistics.stdev(fid)
        print(f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")
        print(f'FID:{avg_fid:.4f} (+/- {var_fid:.4f})')

