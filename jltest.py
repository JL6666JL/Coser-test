@torch.no_grad()
def get_input(self, batch, k=None, return_first_stage_outputs=False, force_c_encode=False,
            cond_key=None, return_original_cond=False, bs=None, val=False, text_cond=[''], return_gt=False, resize_lq=True):

"""Degradation pipeline, modified from Real-ESRGAN:
https://github.com/xinntao/Real-ESRGAN
"""

if not hasattr(self, 'jpeger'):
    jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
if not hasattr(self, 'usm_sharpener'):
    usm_sharpener = USMSharp().cuda()  # do usm sharpening

im_gt = batch['gt'].cuda()  # torch.Size([4, 3, 512, 512])
# True
if self.use_usm:
    im_gt = usm_sharpener(im_gt)
im_gt = im_gt.to(memory_format=torch.contiguous_format).float() # torch.Size([4, 3, 512, 512])
im_ref = batch['ref'].cuda()
if self.use_usm:
    im_ref = usm_sharpener(im_ref)
im_ref = im_ref.to(memory_format=torch.contiguous_format).float()

im_segs = batch['segs']
if self.use_usm:
    im_segs = im_segs.permute(1,0,2,3,4)
    for i in range(im_segs.shape[0]):
        im_segs[i] = usm_sharpener(im_segs[i])
    im_segs = im_segs.permute(1,0,2,3,4)
im_segs = im_segs.to(memory_format=torch.contiguous_format).float()
im_masks = batch['masks']
im_masks = im_masks.to(memory_format=torch.contiguous_format).float()
segs_num = batch['segs_num']

kernel1 = batch['kernel1'].cuda()
kernel2 = batch['kernel2'].cuda()
sinc_kernel = batch['sinc_kernel'].cuda()

ori_h, ori_w = im_gt.size()[2:4]

# ----------------------- The first degradation process ----------------------- #
# blur
out = filter2D(im_gt, kernel1)
# random resize
updown_type = random.choices(
        ['up', 'down', 'keep'],
        self.configs.degradation['resize_prob'],
        )[0]
if updown_type == 'up':
    scale = random.uniform(1, self.configs.degradation['resize_range'][1])
elif updown_type == 'down':
    scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
else:
    scale = 1
mode = random.choice(['area', 'bilinear', 'bicubic'])
out = F.interpolate(out, scale_factor=scale, mode=mode)
# add noise
gray_noise_prob = self.configs.degradation['gray_noise_prob']
if random.random() < self.configs.degradation['gaussian_noise_prob']:
    out = random_add_gaussian_noise_pt(
        out,
        sigma_range=self.configs.degradation['noise_range'],
        clip=True,
        rounds=False,
        gray_prob=gray_noise_prob,
        )
else:
    out = random_add_poisson_noise_pt(
        out,
        scale_range=self.configs.degradation['poisson_scale_range'],
        gray_prob=gray_noise_prob,
        clip=True,
        rounds=False)
# JPEG compression
jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
out = jpeger(out, quality=jpeg_p)

# ----------------------- The second degradation process ----------------------- #
# blur
if random.random() < self.configs.degradation['second_blur_prob']:
    out = filter2D(out, kernel2)
# random resize
updown_type = random.choices(
        ['up', 'down', 'keep'],
        self.configs.degradation['resize_prob2'],
        )[0]
if updown_type == 'up':
    scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
elif updown_type == 'down':
    scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
else:
    scale = 1
mode = random.choice(['area', 'bilinear', 'bicubic'])
out = F.interpolate(
        out,
        size=(int(ori_h / self.configs.sf * scale),
                int(ori_w / self.configs.sf * scale)),
        mode=mode,
        )
# add noise
gray_noise_prob = self.configs.degradation['gray_noise_prob2']
if random.random() < self.configs.degradation['gaussian_noise_prob2']:
    out = random_add_gaussian_noise_pt(
        out,
        sigma_range=self.configs.degradation['noise_range2'],
        clip=True,
        rounds=False,
        gray_prob=gray_noise_prob,
        )
else:
    out = random_add_poisson_noise_pt(
        out,
        scale_range=self.configs.degradation['poisson_scale_range2'],
        gray_prob=gray_noise_prob,
        clip=True,
        rounds=False,
        )

# JPEG compression + the final sinc filter
# We also need to resize images to desired sizes. We group [resize back + sinc filter] together
# as one operation.
# We consider two orders:
#   1. [resize back + sinc filter] + JPEG compression
#   2. JPEG compression + [resize back + sinc filter]
# Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
if random.random() < 0.5:
    # resize back + the final sinc filter
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
            out,
            size=(ori_h // self.configs.sf,
                    ori_w // self.configs.sf),
            mode=mode,
            )
    out = filter2D(out, sinc_kernel)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)
else:
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)
    # resize back + the final sinc filter
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
            out,
            size=(ori_h // self.configs.sf,
                    ori_w // self.configs.sf),
            mode=mode,
            )
    out = filter2D(out, sinc_kernel)

# clamp and round
im_lq = torch.clamp(out, 0, 1.0)
im_segs = torch.clamp(im_segs, 0, 1.0)  # im_segs已经被归一化了

# random crop
gt_size = self.configs.degradation['gt_size']
# 在原始配置中gt_size=512，self.configs.sf=4.gt和lq也是根据这两个函数生成的，裁剪其实没有起到作用
# 如果该函数起到了裁剪的作用，那么segs和masks其实也需要裁剪。但目前没有起到实际的裁剪作用，就先不处理segs和masks
im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, self.configs.sf)

self.lq, self.gt, self.ref = im_lq, im_gt, im_ref

self.lq_clip = self.lq.detach().clone().contiguous()    #因为后面self.lq要采样生成后验分布，所以这里需要深拷贝

self.segs = im_segs.detach().clone().contiguous()    #使用成员变量传递参数，而不是在get_input()的返回值中传递
self.masks = im_masks.detach().clone().contiguous()    
self.segs_num = segs_num

if resize_lq:
    self.lq = F.interpolate(
            self.lq,
            size=(self.gt.size(-2),
                    self.gt.size(-1)),
            mode='bicubic',
            )

if random.random() < self.configs.degradation['no_degradation_prob'] or torch.isnan(self.lq).any():
    self.lq = self.gt

# training pair pool
if not val and not self.random_size:
    self._dequeue_and_enqueue()
# sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
self.lq = self.lq*2 - 1.0
self.gt = self.gt*2 - 1.0
self.ref = self.ref*2 - 1.0

if self.random_size:
    self.lq, self.gt = self.randn_cropinput(self.lq, self.gt)

self.lq = torch.clamp(self.lq, -1.0, 1.0)
self.ref = torch.clamp(self.ref, -1.0, 1.0)

x = self.lq
y = self.gt
r = self.ref
if bs is not None:
    x = x[:bs]
    y = y[:bs]
    r = r[:bs]
x = x.to(self.device)
y = y.to(self.device)
r = r.to(self.device)

# z,z_ref,z_gt都是后验分布了，都是从高斯分布中采样获得的
encoder_posterior = self.encode_first_stage(x)
if self.fixed_cond:
    encoder_posterior = encoder_posterior.mode()
z = self.get_first_stage_encoding(encoder_posterior).detach()   # lr

encoder_posterior = self.encode_first_stage(r)
if self.fixed_cond:
    encoder_posterior = encoder_posterior.mode()
z_ref = self.get_first_stage_encoding(encoder_posterior).detach()

encoder_posterior_y = self.encode_first_stage(y)
z_gt = self.get_first_stage_encoding(encoder_posterior_y).detach()

xc = None
if self.use_positional_encodings:
    assert NotImplementedError
    pos_x, pos_y = self.compute_latent_shifts(batch)
    c = {'pos_x': pos_x, 'pos_y': pos_y}

# while len(text_cond) < z.size(0):
#     text_cond.append(text_cond[-1])
# if len(text_cond) > z.size(0):
#     text_cond = text_cond[:z.size(0)]
# assert len(text_cond) == z.size(0)
text_cond = batch["caption"]
self.seg_c = batch['seg_captions']

out = [z, text_cond, z_gt, z_ref]

# 因为print出来，out的长度是4，所以这两个if条件都没成立
if return_first_stage_outputs:
    xrec = self.decode_first_stage(z_gt)
    out.extend([x, self.gt, xrec])
if return_original_cond:
    out.append(xc)

return out

# x,gt,ref都是从对角高斯分布中采样后的后验分布了，c还是文本形式的caption
def forward(self, x, c, gt, ref,*args, **kwargs):
    index = np.random.randint(0, self.num_timesteps, size=x.size(0))
    t = torch.from_numpy(index)
    t = t.to(self.device).long()

    t_ori = torch.tensor([self.ori_timesteps[index_i] for index_i in index])
    t_ori = t_ori.long().to(x.device)

    if self.model.conditioning_key is not None:
        assert c is not None
        if self.cond_stage_trainable:
            c = self.get_learned_conditioning(c)
        else:
            c, tokens = self.cond_stage_model(c)    # [4, 77, 1024] 在原始配置中只跑这一句，把caption经过clip进行编码
        if self.shorten_cond_schedule:  # TODO: drop this option
            print(s)
            tc = self.cond_ids[t].to(self.device)
            c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

    seg_c = []
    for now_c in self.seg_c:
        if self.model.conditioning_key is not None:
            assert now_c is not None
            if self.cond_stage_trainable:
                now_c = self.get_learned_conditioning(now_c)
            else:
                now_c, tokens = self.cond_stage_model(now_c)    # [4, 77, 1024] 在原始配置中只跑这一句，把caption经过clip进行编码
            if self.shorten_cond_schedule:  # TODO: drop this option
                print(s)
                tc = self.cond_ids[t].to(self.device)
                now_c = self.q_sample(x_start=now_c, t=tc, noise=torch.randn_like(now_c.float()))
        seg_c.append(now_c)
    seg_c = torch.stack(seg_c)  # [30, 4, 77, 1024]
    seg_c = seg_c.permute(1,0,2,3).contiguous()

    # clip image embedding
    image_c = self.pre_sr_model(self.lq_clip)   # 先用垃圾一点的预训练超分模型超分一下，self.lq_clip是LR image
    gt_resize = F.interpolate((self.gt+1.0)/2, size=(image_c.size(-2), image_c.size(-1)), mode='bicubic')
    pre_sr_loss = (gt_resize - image_c).abs().mean()
    
    # cond_stage_model是FrozenOpenCLIPImageTokenEmbedder，也就是CLIP，用CLIP的视觉编码器(ViT)进行编码
    image_features = self.cond_stage_model.encode_with_vision_transformer(self.clip_transfrom(image_c)) # [4, 257, 1024]

    seg_features = []
    for i in range(self.segs.shape[0]):
        seg_features.append(self.cond_stage_model.encode_with_vision_transformer(self.clip_transfrom(self.segs[i])))
    seg_features = torch.stack(seg_features)    #[4, 30, 257, 1024]
    
    # cognitive embedding,self.global_adapter_optimize为False
    # cog_emb.shape: [4, 50, 1024], (B,N,C)
    if self.global_adapter_optimize:
        cog_emb, clip_adapt_loss = self.global_adapter(image_features, c, tokens)
    else:
        cog_emb = self.global_adapter(image_features)
        clip_adapt_loss = 0

    # 如果drop_，c是空的
    if random.random() < self.drop_rate:
        text = ['' for _ in range(x.size(0))]
        c, _ = self.cond_stage_model(text)

    condition_dic = {'prompt_emb': c, 'lr_prompt_emb': cog_emb, 'reference': ref, 'seg_prompt_emb': seg_c, 'seg_img_emb': seg_features}

    return self.p_losses(gt, condition_dic, t, t_ori, x, pre_sr_loss, clip_adapt_loss, *args, **kwargs)
