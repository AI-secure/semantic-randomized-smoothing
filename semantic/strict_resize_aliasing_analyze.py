
import setproctitle
import os
import math

# evaluate a smoothed classifier on a dataset
import argparse
# import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from semantic.core import StrictRotationSmooth
from time import time
import torch
import datetime
from architectures import get_architecture
from semantic.transformers import ResizeTransformer
from semantic.transforms import visualize


EPS = 1e-5

def diff(a, b):
    dif = torch.sum(torch.abs(a - b))
    print('diff:', dif)
    return diff

def calc_dist_map(canopy):
    _, h, w = canopy.shape
    cy, cx = (h - 1.0) / 2.0, (w - 1.0) / 2.0
    rows = torch.linspace(0.0, h - 1, steps=h)
    cols = torch.linspace(0.0, w - 1, steps=w)
    dist_mat = ((rows - cy) * (rows - cy)).unsqueeze(1) + ((cols - cx) * (cols - cx)).unsqueeze(0)
    dist_mat = torch.sqrt(dist_mat)
    return dist_mat

def get_local_maps(img, sr, sl):

    c, h, w = img.shape
    cy, cx = (h-1) / 2.0, (w-1) / 2.0

    map_maxv = torch.zeros((c, h+1, w+1))
    map_maxv[:, :-1, :-1] = torch.max(map_maxv[:, :-1, :-1], img)
    map_maxv[:, +1:, :-1] = torch.max(map_maxv[:, +1:, :-1], img)
    map_maxv[:, :-1, +1:] = torch.max(map_maxv[:, :-1, +1:], img)
    map_maxv[:, +1:, +1:] = torch.max(map_maxv[:, +1:, +1:], img)
    map_minv = torch.zeros((c, h+1, w+1))
    map_minv[:, 1:-1, 1:-1] = img[:, :-1, :-1]
    map_minv[:, 1:-1, 1:-1] = torch.min(map_minv[:, 1:-1, 1:-1], img[:, :-1, +1:])
    map_minv[:, 1:-1, 1:-1] = torch.min(map_minv[:, 1:-1, 1:-1], img[:, +1:, :-1])
    map_minv[:, 1:-1, 1:-1] = torch.min(map_minv[:, 1:-1, 1:-1], img[:, +1:, +1:])
    map_maxd = map_maxv - map_minv

    # brute force for correctness checking
    # t_maxvr_map = torch.zeros_like(img)
    # t_maxdr_map = torch.zeros_like(img)
    # t_maxvl_map = torch.zeros_like(img)
    # t_maxdl_map = torch.zeros_like(img)
    # for i in range(h):
    #     for j in range(w):
    #         nir = (i - cy) / sr + cy
    #         njr = (j - cx) / sr + cx
    #         nil = (i - cy) / sl + cy
    #         njl = (j - cx) / sl + cx
    #         nirf, njrf, nilf, njlf = math.floor(nir), math.floor(njr), math.floor(nil), math.floor(njl)
    #         nir, njr, nil, njl = math.ceil(nir), math.ceil(njr), math.ceil(nil), math.ceil(njl)
    #         nir, njr, nil, njl = int(nir), int(njr), int(nil), int(njl)
    #         if 0 <= nirf and nir <= h-1 and 0 <= njrf and njr <= w-1:
    #             t_maxvr_map[:, i, j] = torch.max(t_maxvr_map[:, i, j], map_maxv[:, nir, njr])
    #             t_maxdr_map[:, i, j] = torch.max(t_maxdr_map[:, i, j], map_maxd[:, nir, njr])
    #         if 0 <= nilf and nil <= h-1 and 0 <= njlf and njl <= w-1:
    #             t_maxvl_map[:, i, j] = torch.max(t_maxvl_map[:, i, j], map_maxv[:, nil, njl])
    #             t_maxdl_map[:, i, j] = torch.max(t_maxdl_map[:, i, j], map_maxd[:, nil, njl])
    # t_maxv_map = torch.max(t_maxvr_map, t_maxvl_map)
    # t_maxd_map = torch.max(t_maxdr_map, t_maxdl_map)
    # brute force part ends

    rows = torch.linspace(0.0, h - 1, steps=h)
    cols = torch.linspace(0.0, w - 1, steps=w)
    nyrs = (rows - cy) / sr + cy
    nxrs = (cols - cx) / sr + cx
    nyr_mat = nyrs.unsqueeze(1).repeat(1, w)
    nxr_mat = nxrs.repeat(h, 1)
    nyls = (rows - cy) / sl + cy
    nxls = (cols - cx) / sl + cx
    nyl_mat = nyls.unsqueeze(1).repeat(1, w)
    nxl_mat = nxls.repeat(h, 1)

    nxl_mat, nxr_mat, nyl_mat, nyr_mat = \
        torch.ceil(nxl_mat).type(torch.LongTensor), torch.ceil(nxr_mat).type(torch.LongTensor), \
        torch.ceil(nyl_mat).type(torch.LongTensor), torch.ceil(nyr_mat).type(torch.LongTensor)

    # handling sr
    il = max(math.ceil(cy * (1.0 - sr)), 0)
    ir = min(math.floor(sr * (h-1) + cy * (1.0 - sr)), h-1)
    jl = max(math.ceil(cx * (1.0 - sr)), 0)
    jr = min(math.floor(sr * (w-1) + cy * (1.0 - sr)), w-1)
    # il = max(math.floor(-sr + cy * (1.0 - sr)) + 1, 0)
    # ir = min(math.floor(sr * h + cy * (1.0 - sr)), h-1)
    # jl = max(math.floor(-sr + cx * (1.0 - sr)) + 1, 0)
    # jr = min(math.floor(sr * w + cx * (1.0 - sr)), w-1)

    maxv_sr_mat = torch.zeros_like(img)
    maxv_sr_mat[:, il: ir+1, jl: jr+1] = torch.gather(map_maxv.reshape(c, (h+1) * (w+1)), dim=1,
                                                      index=(nyr_mat[il: ir + 1, jl: jr + 1] * (w+1) + nxr_mat[il: ir + 1, jl: jr + 1])
                                                      .flatten().repeat(c, 1)).reshape(c, ir-il+1, jr-jl+1)

    maxd_sr_mat = torch.zeros_like(img)
    maxd_sr_mat[:, il: ir+1, jl: jr+1] = torch.gather(map_maxd.reshape(c, (h+1) * (w+1)), dim=1,
                                                      index=(nyr_mat[il: ir + 1, jl: jr + 1] * (w+1) + nxr_mat[il: ir + 1, jl: jr + 1])
                                                      .flatten().repeat(c, 1)).reshape(c, ir-il+1, jr-jl+1)


    # maxv_sr_mat_old = torch.zeros_like(img)
    # maxv_sr_mat_old[:, il: ir+1, jl: jr+1] = torch.gather(
    #     torch.index_select(map_maxv, dim=1, index=nyr_mat[il: ir + 1, jl: jr + 1].flatten()),
    #     dim=2, index=nxr_mat[il: ir + 1, jl: jr + 1].flatten().repeat(c, 1).unsqueeze(2)).reshape(c, ir-il+1, jr-jl+1)
    # maxd_sr_mat_old = torch.zeros_like(img)
    # maxd_sr_mat_old[:, il: ir+1, jl: jr+1] = torch.gather(
    #     torch.index_select(map_maxd, dim=1, index=nyr_mat[il: ir + 1, jl: jr + 1].flatten()),
    #     dim=2, index=nxr_mat[il: ir + 1, jl: jr + 1].flatten().repeat(c, 1).unsqueeze(2)).reshape(c, ir-il+1, jr-jl+1)
    #
    # diff(maxv_sr_mat, maxv_sr_mat_old)
    # diff(maxd_sr_mat, maxd_sr_mat_old)

    # handling sl
    il = max(math.ceil(cy * (1.0 - sl)), 0)
    ir = min(math.floor(sl * (h-1) + cy * (1.0 - sl)), h-1)
    jl = max(math.ceil(cx * (1.0 - sl)), 0)
    jr = min(math.floor(sl * (w-1) + cy * (1.0 - sl)), w-1)
    # il = max(math.floor(-sl + cy * (1.0 - sl)) + 1, 0)
    # ir = min(math.floor(sl * h + cy * (1.0 - sl)), h - 1)
    # jl = max(math.floor(-sl + cx * (1.0 - sl)) + 1, 0)
    # jr = min(math.floor(sl * w + cx * (1.0 - sl)), w - 1)


    maxv_sl_mat = torch.zeros_like(img)
    maxv_sl_mat[:, il: ir+1, jl: jr+1] = torch.gather(map_maxv.reshape(c, (h+1) * (w+1)), dim=1,
                                                      index=(nyl_mat[il: ir + 1, jl: jr + 1] * (w+1) + nxl_mat[il: ir + 1, jl: jr + 1])
                                                      .flatten().repeat(c, 1)).reshape(c, ir-il+1, jr-jl+1)

    maxd_sl_mat = torch.zeros_like(img)
    maxd_sl_mat[:, il: ir+1, jl: jr+1] = torch.gather(map_maxd.reshape(c, (h+1) * (w+1)), dim=1,
                                                      index=(nyl_mat[il: ir + 1, jl: jr + 1] * (w+1) + nxl_mat[il: ir + 1, jl: jr + 1])
                                                      .flatten().repeat(c, 1)).reshape(c, ir-il+1, jr-jl+1)

    # maxv_sl_mat_old = torch.zeros_like(img)
    # maxv_sl_mat_old[:, il: ir+1, jl: jr+1] = torch.gather(
    #     torch.index_select(map_maxv, dim=1, index=nyl_mat[il: ir + 1, jl: jr + 1].flatten()),
    #     dim=2, index=nxl_mat[il: ir + 1, jl: jr + 1].flatten().repeat(c, 1).unsqueeze(2)).reshape(c, ir-il+1, jr-jl+1)
    # maxd_sl_mat_old = torch.zeros_like(img)
    # maxd_sl_mat_old[:, il: ir+1, jl: jr+1] = torch.gather(
    #     torch.index_select(map_maxd, dim=1, index=nyl_mat[il: ir + 1, jl: jr + 1].flatten()),
    #     dim=2, index=nxl_mat[il: ir + 1, jl: jr + 1].flatten().repeat(c, 1).unsqueeze(2)).reshape(c, ir-il+1, jr-jl+1)
    #
    # diff(maxv_sl_mat, maxv_sl_mat_old)
    # diff(maxd_sl_mat, maxd_sl_mat_old)

    ret_maxv = torch.max(maxv_sl_mat, maxv_sr_mat)
    ret_maxd = torch.max(maxd_sl_mat, maxd_sr_mat)

    # print('diff (error checking):', torch.sum(torch.abs(ret_maxv - t_maxv_map)), torch.sum(torch.abs(ret_maxd - t_maxd_map)))

    return ret_maxv, ret_maxd


parser = argparse.ArgumentParser(description='Strict resize lipschitz certify')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("aliasfile", type=str, help='output of alias data')
parser.add_argument("sl", type=float, help="minimum scale ratio")
parser.add_argument("sr", type=float, help="maximum scale ratio")
parser.add_argument("--start", type=int, default=0, help="start before skipping how many examples")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--slice", type=int, default=1000, help="number of angle slices")
parser.add_argument("--subslice", type=int, default=500, help="number of subslices for maximum l2 estimation")
parser.add_argument("--verbstep", type=int, default=10, help="print for how many subslices")
args = parser.parse_args()

if __name__ == '__main__':
    torch.set_num_threads(2)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    # init transformers
    resizer = ResizeTransformer(dataset[0][0], args.sl, args.sr)

    # modify outfile name to distinguish different parts
    if args.start != 0 or args.max != -1:
        args.aliasfile += f'_start_{args.start}_end_{args.max}'

    setproctitle.setproctitle(f'resize_aliasing_{args.dataset}from{args.start}to{args.max}')

    if not os.path.exists(os.path.dirname(args.aliasfile)):
        os.makedirs(os.path.dirname(args.aliasfile))
    f = open(args.aliasfile, 'w')
    print('no.\tmaxl2sqr', file=f, flush=True)

    before_time = time()

    gbl_k = (1.0 / args.sl - 1.0 / args.sr) / (args.slice - 1)
    gbl_c = float(args.slice - 1) / (args.sr / args.sl - 1.0)

    dist_map = calc_dist_map(dataset[0][0])

    # calculating dividing points
    _, h, w = dataset[0][0].shape
    dvdpts = [1.0 - 2.0 * i / (h - 1.0) for i in range(0, math.ceil((h - 1.0) / 2.0))] + \
             [1.0 - 2.0 * i / (w - 1.0) for i in range(0, math.ceil((w - 1.0) / 2.0))]
    dvdpts = torch.unique(torch.sort(torch.tensor(dvdpts))[0])

    for i in range(len(dataset)):

        if i < args.start:
            continue

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i >= args.max >= 0:
            break

        (x, label) = dataset[i]

        print('working on #', i)

        global_max_aliasing = 0.0
        global_before_time = time()
        for j in range(args.slice-1):

            local_sr = 1.0 / (gbl_k * (gbl_c + j))
            local_sl = 1.0 / (gbl_k * (gbl_c + j + 1))

            local_k = (1.0 / local_sl - 1.0 / local_sr) / (args.subslice - 1)
            local_c = float(args.subslice - 1) / (local_sr / local_sl - 1.0)

            max_aliasing = 0.0

            dvdpt = dvdpts[(dvdpts >= local_sl) * (dvdpts <= local_sr)]
            if len(dvdpt) == 0:
                # only need to make sure dvdpt < local_sl
                dvdpt = local_sl - 10.0
            elif len(dvdpt) > 1:
                raise Exception("Containing multiple dividing points, please increase number of slices")
            else:
                dvdpt = dvdpt[0].item()
                print(f'Contain dividing point {dvdpt}')

            base_img_r = resizer.resizer.proc(x, local_sr)
            base_img_l = resizer.resizer.proc(x, local_sl)

            max_v_mapr, max_d_mapr = get_local_maps(x, local_sr, max(local_sl, dvdpt) + EPS)
            L_r = 2.0 * math.sqrt(2) * torch.sum(max_v_mapr * max_d_mapr * dist_map)
            D_r = L_r * local_k / 2.0
            if dvdpt >= local_sl:
                max_v_mapl, max_d_mapl = get_local_maps(x, dvdpt - EPS, local_sl)
                L_l = 2.0 * math.sqrt(2) * torch.sum(max_v_mapl * max_d_mapl * dist_map)
                D_l = L_l * local_k / 2.0

            s_r = local_sr
            alias_k_r = 0.0

            for k in range(args.subslice-1):
                s_l = min(max(1.00 / (local_k * (local_c + k + 1)), local_sl), local_sr)

                x_k_l = resizer.resizer.proc(x, s_l)
                if s_l > dvdpt:
                    alias_k_l = torch.sum((x_k_l - base_img_r) * (x_k_l - base_img_r))
                    D = D_r
                elif s_l < dvdpt:
                    alias_k_l = torch.sum((x_k_l - base_img_l) * (x_k_l - base_img_l))
                    D = D_l
                else:
                    alias_k_l = min(torch.sum((x_k_l - base_img_r) * (x_k_l - base_img_r)), torch.sum((x_k_l - base_img_l) * (x_k_l - base_img_l)))

                if s_l <= dvdpt <= s_r:
                    now_max_alias = max(alias_k_r + L_r * (1.0 / dvdpt - 1.0 / s_r), alias_k_l + L_l * (1.0 / s_l - 1.0 / dvdpt))
                else:
                    now_max_alias = (alias_k_l + alias_k_r) / 2.0 + D
                max_aliasing = max(now_max_alias.item(), max_aliasing)

                s_r = s_l
                alias_k_r = alias_k_l

            global_max_aliasing = max(global_max_aliasing, max_aliasing)
            if j % args.verbstep == 0:
                print(i, f'{j}/{args.slice}', f'[{local_sl:.3f}, {local_sr:.3f}]', max_aliasing, global_max_aliasing, str(datetime.timedelta(seconds=(time() - before_time))))
                before_time = time()

        print(f'{i}\t{global_max_aliasing}\t{str(datetime.timedelta(seconds=(time() - global_before_time)))}', file=f, flush=True)

    f.close()


