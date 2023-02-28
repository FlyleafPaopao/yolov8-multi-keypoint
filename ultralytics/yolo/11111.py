
# from tqdm import tqdm
# s = [[1,5,4], 2, 3, 4, 5.1111111111111111111]
# e = enumerate(s)
# pbar = tqdm(e)
#
# for i,char in pbar:
#     print(i,char)
def build_targets(self, p, targets):
    """build targets方法
        @params p: 网络训练阶段输出，格式[torch.tensor(n, c, h1, w1), torch.tensor(n, c, h2, w2), (n, c, h3, w3)]
        @params targets: 数据标签，dataloader输出，格式(m, 6)，需要注意的是虽然batch = n 但是target的数量要大于等于n，是因为单张图片可能存在多个目标
    """
    # 每个尺度建议框的数量： na
    # 当前batch目标框数量：nt
    na, nt = self.na, targets.shape[0]

    tcls, tbox, indices, anch = [], [], [], []

    # 将标签中归一化后的xywh映射到特征图上的比例： gain
    # 之所以初始化为7，是因为原标签数据包含6个数据，然后还需要在尾部加上建议框id组成(m, 7)结构的标签
    gain = torch.ones(7, device=targets.device)

    # 标签由(m, 6)编程(na, m, 7)结构，在每个元素尾部加入建议框id
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

    # 偏移量：off 参考下图看
    # ----------|--------|--------|
    # |         | (0, -1)|        |
    # ----------|--------|--------|
    # | (-1, 0) | (0, 0) | (1, 0) |
    # ----------|--------|--------|
    # |         | (0, 1) |        |
    # ----------|--------|--------|
    g = 0.5
    off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * g

    # 输出特征图的数量：nl
    for i in range(self.nl):
        # self.anchors：(nl, 3, 2)
        # anchors: (3, 2)
        anchors = self.anchors[i]

        # p[i]：[n, c, h, w]
        # gain：[1, 1, w, h, w, h, 1]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

        # targets: [img_id, cls_id, x_norm, y_norm, w_norm, h_norm, anchor_id]
        # 将标签中归一化后的xywh映射到特征图上
        t = targets * gain

        if nt:
            # anchors[:, None]：(3, 1, 2)
            # t：(3, m, 2)
            # 获取anchor与gt的宽高比值，如果比值超出anchor_t，那么该anchor就会被舍弃，不参与loss计算
            r = t[:, :, 4:6] / anchors[:, None]
            j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']
            t = t[j]

            # 中心点：gxy
            # 反转中心点：gxi
            gxy = t[:, 2:4]
            gxi = gain[[2, 3]] - gxy

            # 距离当前格子左上角较近的中心点，并且不是位于边缘格子内
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            # 距离当前格子右下角较近的中心点，并且不是位于边缘格子内
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T

            # j和l， k和m是互斥的，一个为True，那么另一个必定为False
            # j：(5, m)
            # j：[[all],
            #       [j == True],
            #       [k == True],
            #       [l == True],
            #       [m == True]]
            j = torch.stack((torch.ones_like(j), j, k, l, m))

            # t：(5, m, 5)
            # t[j]：(m', 7)
            t = t.repeat((5, 1, 1))[j]
            # shape：(1, m, 2) + (5, 1, 2) = (5, m, 2)[j] = (m', 2)
            # offsets排列(g = 0.5)：(0, 0), (0.5, 0), (0, 0.5), (-0.5, 0), (0, -0.5)
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # b：img_id
        # c: cls_id
        b, c = t[:, :2].long().T
        # gxy和gwh当前是基于特征图尺寸的数据
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]
        # 注意offsets的排列顺序，做减法，其结果为：(0, 0) + 四选二((-1, 0), (0, -1), (1, 0), (0, 1))
        # gij就是正样本格子的整数部分即索引
        gij = (gxy - offsets).long()
        gi, gj = gij.T

        # 去掉anchor_id
        a = t[:, 6].long()

        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
        # 这里(gxy-gij)的取值范围-0.5 ~ 1.5
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch