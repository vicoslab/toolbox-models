import os

import numpy as np
import torch
from utils import transforms as my_transforms
from torchvision.transforms import InterpolationMode

def get_args(width, height, enable_6dof=False, enable_symmetries=False, use_depth=False):
	num_fields = 3 + (6 if enable_6dof else 2)
	args = dict(

		cuda=True,
		cuda_sync_with_file=False,
		display=True,
		save=True,

		display_per_error_type_gradients=False,

		pretrained_model_path = None,
		pretrained_center_model_path = 'center-model.pth',

		precompute_gt_cache=False,

		train_dataset = {
			'name': 'screw',
			'kwargs': {
				'normalize': False,
				'type': 'train_pbr',
				'scene_id': '000000',
				'fixed_bbox_size': 15, #dataset does not have mask so we need to use this
				'output_single_orientation_only':(False if enable_6dof else 'x'),
				# 'categories': [1,5,6], # should only return ape, can and cat

				'categories': [8], # driller

				'transform': my_transforms.get_transform([
					{
						'name': 'ToTensor',
						'opts': {
							'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask'),
							'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor, torch.ByteTensor),
						}
					},
					{
						'name': 'Resize',
						'opts': {
							'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask'),
							'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.NEAREST),
							'keys_bbox': ('center',),
							'size': (height, width),
						}
					},
					{
						'name': 'RandomHorizontalFlip',
						'opts': {
							'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if use_depth else ()), 'keys_bbox': ('center',),
							'keys_custom_fn' : { 'orientation': lambda x: (np.pi - x + np.pi) % (2 * np.pi) - np.pi},
							# 'keys_custom_fn' : { 'orientation': lambda x: np.atan2(np.sin(x), -np.cos(x))},
							'p': 0.25,
						}
					},
					{
						'name': 'RandomVerticalFlip',
						'opts': {
							'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if use_depth else ()), 'keys_bbox': ('center',),
							'keys_custom_fn' : { 'orientation': lambda x: (2*np.pi - x + np.pi) % (2 * np.pi) - np.pi},
							# 'keys_custom_fn' : { 'orientation': lambda x: np.atan2(-np.sin(x), np.cos(x))},
							'p': 0.25,
						}
					},
					{
						'name': 'RandomCustomRotation',
						'opts': {
							'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if use_depth else ()), 'keys_bbox': ('center',),
							'keys_custom_fn' : { 'orientation': lambda x,angle: (x - np.deg2rad(angle) + np.pi) % (2 * np.pi) - np.pi}, # we subtract angle here because rotations go clockwise, but angles go anti-clockwise
							'resample': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST,
												InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST)  + ((InterpolationMode.BILINEAR, ) if use_depth else ()),
							'angles': list(range(0, 360, 10)),
							'rate':0.25,
						}
					},
					{
						'name': 'ColorJitter',
						'opts': {
							'keys': ('image',), 'p': 0.25,
							'saturation': 0.3, 'hue': 0.3, 'brightness': 0.3, 'contrast':0.3
						}
					},
				]),
				'MAX_NUM_CENTERS':2*1024,
			},

			'centerdir_gt_opts': dict(
				ignore_instance_mask_and_use_closest_center=True,
				center_ignore_px=3,
				skip_gt_center_mask_generate=True,
				MAX_NUM_CENTERS=2*1024,
			),

			'batch_size': 2,
			# 'batch_size': 8,
			'hard_samples_size': 0,
			'hard_samples_selected_min_percent':0.1,
			'workers': 4,
			'shuffle': True,
		}, 

		model = dict(
			name='fpn',
			kwargs= {
				#'backbone': 'efficientnet-b0',
				#'backbone': 'resnet101',
				# 'backbone': 'resnet50',
				'backbone': 'tu-convnext_base',
				#'backbone': 'resnet18',
				# 'num_classes': [3, 1],
				'num_classes': [num_fields, 1],
				'use_custom_fpn':True,
				'add_output_exp': False,
				'fpn_args': {
					'upsampling':4, # required for ConvNext architectures
					#'decoder_dropout':0
					'decoder_segmentation_head_channels':64,
					'depth_mean': None,
					'depth_std': None,
				},
				'init_decoder_gain': 0.1
			},
			# 'name': 'branched_erfnet',
			# 'kwargs': {
			#    'num_classes': [3, 1]
			# }
			optimizer='Adam',
			# resnet50-depth4 version
			# lr=1e-4,
			# n_epochs=1000,
			# weight_decay=1e-4,
			# resnet50-depth4-no-w_decay
			# lr=2.5e-4,
			# last good version ---
			lr=1e-4,
			weight_decay=0,

		),
		center_model=dict(
			name='CenterOrientationEstimator',
			kwargs=dict(
				# use vector magnitude as mask instead of regressed mask
				use_magnitude_as_mask=False,
				# thresholds for conv2d processing
				local_max_thr=0.1, mask_thr=0.01, hough_thr=5000, exclude_border_px=0,
				# thresholds for mask suppression using hough estimation (hough_mask_thr == thr applied to magnitude)
				suppression_by_mask=True, hough_mask_thr=0.2,
				# number of jumps/hops for mask estimation from votes
				hough_num_hops=5,
				use_dilated_nn=True,
				ignore_cls_prediction=True,
				ignore_polar_magnitude=True,
				dilated_nn_args=dict(
					# single scale version (nn6)
					inner_ch=16,
					inner_kernel=3,
					dilations=[1, 4, 8, 12],
					freeze_learning=True,
					gradpass_relu=False,
					# version with leaky relu
					leaky_relu=False,
					# input check
					# use_polar_radii=False,
					use_centerdir_radii = False,
					use_centerdir_magnitude = False,
					use_cls_mask = False
				),
				allow_input_backprop=False,
				backprop_only_positive=False, #True,
				augmentation=False, # cannot use during training of polar prediction since it will interfere with it
				augmentation_kwargs=dict(
					occlusion_probability=0.75,
					occlusion_type='circle',
					occlusion_distance_type='larger', #random
					occlusion_center_jitter_probability=0.5,
					occlusion_center_jitter_relative_size=0.4,
					gaussian_noise_probability=0.25,
					gaussian_noise_blur_sigma=3,
					gaussian_noise_std_polar=[0.1,2.0],
					gaussian_noise_std_mask=[0.1,2.0]
				),
				scale_r=1.0,  # 1024
				scale_r_gt=1024,  # 1
				use_log_r=True,
				use_log_r_base='10',
				enable_6dof=enable_6dof,
			),
			optimizer='Adam',
			lr=0,#1e-4,
			weight_decay=0,
		),

		# loss options
		loss_type='OrientationLoss',
		loss_opts={
			'num_vector_fields': num_fields,
			'foreground_weight': 1,  # 1000,
			# Adding loss at bg pixels for center predictions
			'no_instance_loss': True,  # MUST be True to ignore instance mask
			# 'no_instance_loss': False,  # MUST be True to ignore instance mask

			# Disable cls loss since we assume we do not have instance masks for classification any ways
			# ALSO SET BELOW enable_cls_loss=false
			'cls_no_loss': True,

			# Disable instance weighting since we do not have correct instance masks (lets assume we have small fixed around center!)
			'cls_instance_weighted': True,
			'centerdir_instance_weighted': True,

			# INSTEAD use loss weighting by distance to center point using gaussian
			'loss_weighted_by_distance_gauss': 0,  # sigma for converting R distance to weight (3 -> 0.1 at 40 pix distance)

			# Using hinge loss instead of mean-square-error
			'cls_loss': 'l1',
			'regression_loss': 'l1',

			'magnitude_regularization': None,

			# anything related to instance_mask should not be used
			'num_hard_negatives': 50,
			'hard_negatives_center_mask': 15,

			'extend_instance_mask_weights': False,  # 17,
			'extend_instance_mask_as_hard_negative': False,
			'use_instance_mask_iou_weight': False,
			'instance_mask_iou_only_fp': False,  # This will not mark as false positives pixels where instance > 0
			'instance_mask_iou_ignore_tp': False,  # this will ignore  groundturth centers from instance_mask callculation
			'instance_mask_iou_include_fn': False,
			# this will add false-negative centers only (based on groundtruth region)

			# 'per_error_type_losses': dict(
			#    FP='l1',
			#    FN='l1',
			#    TP=dict(type='smoothL1', args={'beta':2,'pow':4}),
			#    TN=dict(type='smoothL1', args={'beta':2,'pow':4}),
			# ),
			'use_log_r': True,
			'use_log_r_base': '10',

			'learnable_center_est':False,
			'learnable_center_loss':'l1',
			'learnable_center_ignore_negative_gradient': False,
			'learnable_center_fp_threshold': 0.1,
			'learnable_center_with_instance_norm': False,
			'learnable_center_positive_area_radius': 5,

			'enable_centerdir_loss': True,
			'enable_cls_loss': False,

			'border_weight': 1.0,
			'border_weight_px': 0,
			'orientation_args': dict(
				enable=True,
				no_instance_loss=False,
				regression_loss='l1',
				enable_6dof=enable_6dof,
				symmetries=[6,0,0] if enable_6dof and enable_symmetries else None,
			)
	},
		# loss_w={
		# 	'w_inst': 1,
		# 	'w_var': 1,
		# 	'w_seed': 1,
		# 	'w_cls': 1,
		# 	'w_r': 1,
		# 	'w_cos': 1,
		# 	'w_sin': 1,
		# 	'w_magnitude': 1,
		# 	'w_cent': 0.1,
		# 	'w_orientation': 4,
		# },
		loss_w={
			'w_r': 1,
			'w_cos': 1,
			'w_sin': 1,
			'w_cent': 0.1,
			'w_orientation': 1,
		},

	)

	# Original scheduler used by SpatialEmbedding method
	args['lambda_scheduler_fn']=lambda _args: (lambda epoch: pow((1-((epoch)/_args['n_epochs'])), 0.9))
	#args['lambda_scheduler_fn']=lambda _args: (lambda epoch: 1.0) # disabled

	args['model']['lambda_scheduler_fn'] = args['lambda_scheduler_fn']
	args['center_model']['lambda_scheduler_fn'] = lambda _args: (lambda epoch: pow((1-((epoch)/_args['n_epochs'])), 0.9) if epoch > 1 else 0)
	return args
