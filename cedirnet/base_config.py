import os

import torch
from utils import transforms as my_transforms
from torchvision.transforms import InterpolationMode

ENABLE_6DOF = False
# ENABLE_6DOF = False
NUM_FIELDS = 3 + (6 if ENABLE_6DOF else 2)
# ENABLE_SYMMETRIES = True
ENABLE_SYMMETRIES = False
SYMMETRIES = [6,0,0]

dof_name = '6dof' if ENABLE_6DOF else ''
dof_name+='_symmetry' if ENABLE_SYMMETRIES else ''

args = dict(

	cuda=True,
	cuda_sync_with_file=False,
	display=True,
	# display=False,
	# display_it=100,
	display_it=1,
	# display_it=5,

	save=True,
	save_interval=4,

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
			'output_single_orientation_only':(False if ENABLE_6DOF else 'x'),
			# 'categories': [1,5,6], # should only return ape, can and cat

			'categories': [8], # driller

			'transform': my_transforms.get_transform([
				# for training without augmentation (same as testing)

				{
					'name': 'Resize',
					'opts': {
						# 'keys': ('image', 'instance', 'label', 'difficult'),
						# 'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST),
						'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask'),
						'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.NEAREST),
						'keys_bbox': ('center',),

						# 'size': (336, 512),
						'size': (480, 640),
						# 'size': (512, 768),
					}
				},
				# {
				# 	'name': 'Background',
				# 	'opts': {
				# 		# 'p' : 0.5
				# 		'p' : 0.0,
				# 		# 'p' : 1.0
				# 		'bg_dir': os.path.join(STORAGE_DIR, 'datasets/PASCAL/VOC2012/JPEGImages/'),
				# 		'bg_images': os.path.join(STORAGE_DIR, 'datasets/PASCAL/VOC2012/ImageSets/Main/diningtable_trainval.txt'),

				# 	}
				# },
				# for training with random augmentation
				# {
				#     'name': 'RandomGaussianBlur',
				#     'opts': {
				#         'keys': ('image',),
				#         'rate': 0.5, 'sigma': [0.5, 2]
				#     }
				# },
				# {
				# 	'name': 'RandomHorizontalFlip',
				# 	'opts': {
				# 		'keys': ('image', 'instance', 'label', 'ignore', 'orientation'), 'keys_bbox': ('center',),
				# 		'p': 0.5,
				# 	}
				# },
				# {
				# 	'name': 'RandomVerticalFlip',
				# 	'opts': {
				# 		'keys': ('image', 'instance', 'label', 'ignore', 'orientation'), 'keys_bbox': ('center',),
				# 		'p': 0.5,
				# 	}
				# },
				
				# {
				# 'name': 'RandomRotation',
				# 'opts': {
				# 	# 'keys': ('image'),
				# 	'keys': ('image', 'instance', 'label', 'ignore', 'center'),
				# 	# 'rng': None,
				# 	# 'degrees' : 180
				# 	# 'degrees' : (0,30)
				# 	# 'degrees' : (30,60)
				# 	'degrees' : (0,180)
				# }
				# },
				# {
				#     'name': 'RandomResize',
				#     'opts': {
				#         'keys': ('image', 'instance', 'label', 'ignore'), 'keys_bbox': ('center',),
				#         'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST),
				#         'scale_range': [0.5,1.5]
				#     }
				# },
				# {
				#     'name': 'RandomCrop',
				#     'opts': {
				#         'keys': ('image', 'instance', 'label', 'ignore'), 'keys_bbox': ('center',),
				#         # 'size': (256,256),
				#         # 'size': (128,128),
				#         'size': (512,512),
				#         # 'size': (1024,1024),
				#         'pad_if_needed': True
				#     }
				# },
				# {
				# 	'name': 'ColorJitter',
				# 	'opts': {
				# 		'keys': ('image',), 'p': 0.5,
				# 		'saturation': 0.2, 'hue': 0.2, 'brightness': 0.2, 'contrast':0.2
				# 	}
				# },
				{
					'name': 'ToTensor',
					'opts': {
						# 'keys': ('image', 'instance', 'label', 'ignore'),
						# 'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor),
						'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask'),
						'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor, torch.ByteTensor),
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
			'num_classes': [NUM_FIELDS, 1],
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
			enable_6dof=ENABLE_6DOF,
		),
		optimizer='Adam',
		lr=0,#1e-4,
		weight_decay=0,
	),

	# loss options
	loss_type='OrientationLoss',
	loss_opts={
        'num_vector_fields': NUM_FIELDS,
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
			enable_6dof=ENABLE_6DOF,
			symmetries=SYMMETRIES if ENABLE_6DOF and ENABLE_SYMMETRIES else None,
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
