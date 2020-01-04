
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
from semantic.transformers import RotationTransformer
from semantic.transforms import visualize


def rotate(img, angle, mask):
    c, h, w = img.shape
    out = torch.zeros_like(img)
    cy, cx = (h-1) / 2.0, (w-1) / 2.0
    for i in range(h):
        for j in range(w):
            if mask[0][i][j]:
                alpha = math.atan2(i-cy, j-cx)
                dist = math.sqrt((i-cy)*(i-cy) + (j-cx)*(j-cx))
                beta = alpha + angle * math.pi / 180.0
                ny = cy + dist * math.sin(beta)
                nx = cx + dist * math.cos(beta)
                nyl, nxl = math.floor(ny), math.floor(nx)
                nyr, nxr = nyl + 1, nxl + 1
                for k in range(c):
                    out[k][i][j] = img[k][nyl][nxl] * (1.0 - (ny-nyl)) * (1.0 - (nx-nxl)) + \
                                   img[k][nyl][nxr] * (1.0 - (ny-nyl)) * (1.0 - (nxr-nx)) + \
                                   img[k][nyr][nxl] * (1.0 - (nyr-ny)) * (1.0 - (nx-nxl)) + \
                                   img[k][nyr][nxr] * (1.0 - (nyr-ny)) * (1.0 - (nxr-nx))
    return out


def get_lipschitz_bound(canopy, mask):
    c, h, w = canopy.shape
    ans = 0
    cy, cx = (h-1) / 2.0, (w-1) / 2.0
    for i in range(h):
        for j in range(w):
            if mask[0][i][j]:
                dist = math.sqrt((i-cy)*(i-cy) + (j-cx)*(j-cx))
                ans += 2.0 * math.sqrt(2) * dist * c
    # rad to deg
    ans = ans * math.pi / 180.0
    return ans

# based on 2 \sqrt 2 maxV maxD L
def get_finer_lipschitz_bound(img, mask, anglel, angler):

    c, h, w = img.shape
    ans = 0.0
    radl, radr = anglel * math.pi / 180.0, angler * math.pi / 180.0
    cy, cx = (h-1) / 2.0, (w-1) / 2.0
    mask_l = mask.type(torch.LongTensor)

    # preprocess
    # for performance, fxxk python's slow loop, have to unfold manually

    # map_maxv, map_maxd = torch.zeros_like(img), torch.zeros_like(img)
    # for i in range(h-1):
    #     for j in range(w-1):
    #         map_maxv[:, i, j] = img[:, i:(i+2), j:(j+2)].max(dim=2)[0].max(dim=1)[0]
    #         map_maxd[:, i, j] = map_maxv[:, i, j] - img[:, i:(i+2), j:(j+2)].min(dim=2)[0].min(dim=1)[0]

    map_maxv = img.clone().detach()
    map_maxv[:, :-1, :] = torch.max(map_maxv[:, :-1, :], img[:, 1:, :])
    map_maxv[:, :, :-1] = torch.max(map_maxv[:, :, :-1], img[:, :, 1:])
    map_maxv[:, :-1, :-1] = torch.max(map_maxv[:, :-1, :-1], img[:, 1:, 1:])
    map_minv = img.clone().detach()
    map_minv[:, :-1, :] = torch.min(map_minv[:, :-1, :], img[:, 1:, :])
    map_minv[:, :, :-1] = torch.min(map_minv[:, :, :-1], img[:, :, 1:])
    map_minv[:, :-1, :-1] = torch.min(map_minv[:, :-1, :-1], img[:, 1:, 1:])
    map_maxd = map_maxv - map_minv

    # map_nb_maxv, map_nb_maxd = torch.zeros_like(img), torch.zeros_like(img)
    # for i in range(h):
    #     for j in range(w):
    #         map_nb_maxv[:, i, j] = map_maxv[:, max(i-1,0): min(i+2,h), max(j-1,0): min(j+2,w)].max(dim=2)[0].max(dim=1)[0]
    #         map_nb_maxd[:, i, j] = map_maxd[:, max(i-1,0): min(i+2,h), max(j-1,0): min(j+2,w)].max(dim=2)[0].max(dim=1)[0]

    # map_nb_maxv = map_maxv.clone().detach()
    # map_nb_maxv[:, :-1, :-1] = torch.max(map_nb_maxv[:, :-1, :-1], map_maxv[:, 1:, 1:])
    # map_nb_maxv[:, :-1, :] = torch.max(map_nb_maxv[:, :-1, :], map_maxv[:, 1:, :])
    # map_nb_maxv[:, :-1, 1:] = torch.max(map_nb_maxv[:, :-1, 1:], map_maxv[:, 1:, :-1])
    # map_nb_maxv[:, :, :-1] = torch.max(map_nb_maxv[:, :, :-1], map_maxv[:, :, 1:])
    # map_nb_maxv[:, :, 1:] = torch.max(map_nb_maxv[:, :, 1:], map_maxv[:, :, :-1])
    # map_nb_maxv[:, 1:, :-1] = torch.max(map_nb_maxv[:, 1:, :-1], map_maxv[:, :-1, 1:])
    # map_nb_maxv[:, 1:, :] = torch.max(map_nb_maxv[:, 1:, :], map_maxv[:, :-1, :])
    # map_nb_maxv[:, 1:, 1:] = torch.max(map_nb_maxv[:, 1:, 1:], map_maxv[:, :-1, :-1])

    map_nb_maxd = map_maxd.clone().detach()
    map_nb_maxd[:, :-1, :-1] = torch.max(map_nb_maxd[:, :-1, :-1], map_maxd[:, 1:, 1:])
    map_nb_maxd[:, :-1, :] = torch.max(map_nb_maxd[:, :-1, :], map_maxd[:, 1:, :])
    map_nb_maxd[:, :-1, 1:] = torch.max(map_nb_maxd[:, :-1, 1:], map_maxd[:, 1:, :-1])
    map_nb_maxd[:, :, :-1] = torch.max(map_nb_maxd[:, :, :-1], map_maxd[:, :, 1:])
    map_nb_maxd[:, :, 1:] = torch.max(map_nb_maxd[:, :, 1:], map_maxd[:, :, :-1])
    map_nb_maxd[:, 1:, :-1] = torch.max(map_nb_maxd[:, 1:, :-1], map_maxd[:, :-1, 1:])
    map_nb_maxd[:, 1:, :] = torch.max(map_nb_maxd[:, 1:, :], map_maxd[:, :-1, :])
    map_nb_maxd[:, 1:, 1:] = torch.max(map_nb_maxd[:, 1:, 1:], map_maxd[:, :-1, :-1])

    # main
    # tensor accelerated

    # t_v = torch.zeros_like(img[0])
    # t_nyl_mat, t_nxl_mat, t_nyr_mat, t_nxr_mat = \
    #     torch.zeros_like(img[0]).type(torch.LongTensor), torch.zeros_like(img[0]).type(torch.LongTensor), \
    #     torch.zeros_like(img[0]).type(torch.LongTensor), torch.zeros_like(img[0]).type(torch.LongTensor)
    # for i in range(h):
    #     for j in range(w):
    #         if mask_l[0][i][j]:
    #             dist = math.sqrt((i-cy)*(i-cy) + (j-cx)*(j-cx))
    #             # # now does not consider margin
    #             # margin = dist * (1.0 - math.cos((radr - radl)/2.0))
    #             alpha = math.atan2(i-cy, j-cx)
    #             betal = alpha + radl
    #             betar = alpha + radr
    #             nyl, nxl = cy + dist * math.sin(betal), cx + dist * math.cos(betal)
    #             nyr, nxr = cy + dist * math.sin(betar), cx + dist * math.cos(betar)
    #
    #             nyl, nxl, nyr, nxr = math.floor(nyl), math.floor(nxl), math.floor(nyr), math.floor(nxr)
    #
    #             t_nyl_mat[i][j], t_nxl_mat[i][j], t_nyr_mat[i][j], t_nxr_mat[i][j] = nyl, nxl, nyr, nxr
    #
    #             t_v[i][j] = dist * \
    #                    torch.sum(
    #                        torch.max(map_nb_maxv[:, nyl, nxl], map_nb_maxv[:, nyr, nxr]) *
    #                        torch.max(map_nb_maxd[:, nyl, nxl], map_nb_maxd[:, nyr, nxr])
    #                    )
    #             ans += t_v[i][j]

    rows = torch.linspace(0.0, h-1, steps=h)
    cols = torch.linspace(0.0, w-1, steps=w)
    dist_mat = ((rows - cy) * (rows - cy)).unsqueeze(1) + ((cols - cx) * (cols - cx)).unsqueeze(0)
    dist_mat = torch.sqrt(dist_mat)
    rows_mat = rows.unsqueeze(1).repeat(1, w)
    cols_mat = cols.repeat(h, 1)
    alpha_mat = torch.atan2(rows_mat - cy, cols_mat - cx)
    betal_mat = alpha_mat + radl
    betar_mat = alpha_mat + radr
    nyl_mat, nxl_mat = dist_mat * torch.sin(betal_mat) + cy, dist_mat * torch.cos(betal_mat) + cx
    nyr_mat, nxr_mat = dist_mat * torch.sin(betar_mat) + cy, dist_mat * torch.cos(betar_mat) + cx
    nyl_mat, nxl_mat = torch.floor(nyl_mat).type(torch.LongTensor), torch.floor(nxl_mat).type(torch.LongTensor)
    nyr_mat, nxr_mat = torch.floor(nyr_mat).type(torch.LongTensor), torch.floor(nxr_mat).type(torch.LongTensor)
    torch.clamp_(nyl_mat, min=0, max=h-1)
    torch.clamp_(nyr_mat, min=0, max=h-1)
    torch.clamp_(nxl_mat, min=0, max=w-1)
    torch.clamp_(nxr_mat, min=0, max=w-1)

    nyl_cell, nxl_cell, nyr_cell, nxr_cell = torch.flatten(nyl_mat), torch.flatten(nxl_mat), torch.flatten(nyr_mat), torch.flatten(nxr_mat)
    # maxv_pl_mat = torch.gather(torch.index_select(map_nb_maxv, dim=1, index=nyl_cell), dim=2, index=nxl_cell.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
    # maxv_pr_mat = torch.gather(torch.index_select(map_nb_maxv, dim=1, index=nyr_cell), dim=2, index=nxr_cell.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
    maxd_pl_mat = torch.gather(torch.index_select(map_nb_maxd, dim=1, index=nyl_cell), dim=2, index=nxl_cell.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
    maxd_pr_mat = torch.gather(torch.index_select(map_nb_maxd, dim=1, index=nyr_cell), dim=2, index=nxr_cell.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)

    # p_v = (torch.max(maxv_pl_mat, maxv_pr_mat) * torch.max(maxd_pl_mat, maxd_pr_mat)).sum(dim=0) * dist_mat * mask
    # ans += torch.sum(p_v)

    # discrepancy check
    # print(torch.sum(torch.abs(t_v - p_v)))

    # rad to deg
    # ans = ans * 2.0 * math.sqrt(2) * math.pi / 180.0
    return None, torch.max(maxd_pl_mat, maxd_pr_mat), dist_mat


parser = argparse.ArgumentParser(description='Strict rotation certify')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("aliasfile", type=str, help='output of alias data')
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--slice", type=int, default=1000, help="number of angle slices")
parser.add_argument("--subslice", type=int, default=500, help="number of subslices for maximum l2 estimation")
parser.add_argument("--partial", type=float, default=180.0, help="only contain +-angle maximum aliasing")
parser.add_argument("--verbstep", type=int, default=100, help="print for how many subslices")
args = parser.parse_args()

if __name__ == '__main__':
    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    # init transformers
    rotationT = RotationTransformer(dataset[0][0])

    if not os.path.exists(os.path.dirname(args.aliasfile)):
        os.makedirs(os.path.dirname(args.aliasfile))
    f = open(args.aliasfile, 'w')
    print('no.\tmaxl2sqr', file=f, flush=True)

    before_time = time()

    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        print('working on #', i)

        global_max_aliasing = 0.0
        d = 360.0 / (args.slice * args.subslice)
        radd = math.pi / (args.slice * args.subslice)
        for j in range(args.slice):
            max_aliasing = 0.0

            base_ang = 360.0 * j / args.slice

            if 360.0 * j / args.slice > args.partial and 360.0 - 360.0 * (j + 1) / args.slice > args.partial:
                continue

            base_img = rotationT.rotation_adder.proc(x, base_ang)
            _, max_d_map, dist_mat = get_finer_lipschitz_bound(x, rotationT.rotation_adder.mask, base_ang, base_ang + 360.0 / args.slice)

            max_d2_sum = torch.sum(max_d_map * max_d_map)

            ang_l = base_ang
            x_k_l = rotationT.rotation_adder.proc(x, ang_l)
            alias_k_l = torch.sum((x_k_l - base_img) * (x_k_l - base_img))

            term_A = 2.0 * radd * radd * torch.sum(dist_mat * dist_mat * max_d_map * max_d_map * rotationT.rotation_adder.mask)
            interm_mat = dist_mat * max_d_map * rotationT.rotation_adder.mask

            for k in range(args.subslice):
                ang_r = base_ang + (k+1) * d
                x_k_r = rotationT.rotation_adder.proc(x, ang_r)
                alias_k_r = torch.sum((x_k_r - base_img) * (x_k_r - base_img))

                # now_max_alias = (alias_k_l + alias_k_r) / 2.0 + (d * L) / 2.0

                now_max_alias_new = term_A + \
                                    max(2.0 * math.sqrt(2) * radd * torch.sum(interm_mat * x_k_l) + alias_k_l,
                                        2.0 * math.sqrt(2) * radd * torch.sum(interm_mat * x_k_r) + alias_k_r)
                # print(now_max_alias, now_max_alias_new)
                # print(now_max_alias_new)

                max_aliasing = max(now_max_alias_new.item(), max_aliasing)

                ang_l = ang_r
                x_k_l = x_k_r
                alias_k_l = alias_k_r

            global_max_aliasing = max(global_max_aliasing, max_aliasing)
            if j % args.verbstep == 0:
                print(i, f'{j}/{args.slice}', max_aliasing, global_max_aliasing, str(datetime.timedelta(seconds=(time() - before_time))))
                before_time = time()

        print(f'{i}\t{global_max_aliasing}', file=f, flush=True)

    f.close()

    # # debug: compare manual rotation and library rotation
    # # now they are totally equal, means the library rotation is truly bi-linear
    # for i in range(10):
    #     img = dataset[i][0]
    #     angle = rotationT.rotation_adder.gen_param()
    #     my_out = rotate(img, angle, rotationT.rotation_adder.mask)
    #     my_out = rotationT.rotation_adder.masking(my_out)
    #     lib_out = rotationT.rotation_adder.proc(img, angle)
    #     print(i, angle, torch.sum((my_out - lib_out) * (my_out - lib_out)))
    #     visualize(my_out, f'test/test/transform/{args.dataset}/{args.split}/manual-rotation/{i}-{int(angle)}-man.bmp')
    #     visualize(lib_out, f'test/test/transform/{args.dataset}/{args.split}/manual-rotation/{i}-{int(angle)}-lib.bmp')
