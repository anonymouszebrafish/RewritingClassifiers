import os, cox
import torch as ch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import helpers.context_helpers as coh
import helpers.load_helpers as lh
import helpers.gen_helpers as gh
import helpers.grid_helpers as grid

from cox.store import Store

parser = ArgumentParser()

parser.add_argument('--dataset-name', type=str, default='ImageNet')
parser.add_argument('--out-dir', type=str, default='/tmp/output')
parser.add_argument('--expt-name', type=str, 
                    default='config')
parser.add_argument('--concepts', type=str, default='COCO')
parser.add_argument('--concept-dir', type=str)
parser.add_argument('--styles', type=list)
parser.add_argument('--style-dir', type=str)

parser.add_argument('--arch', type=str)
parser.add_argument('--epsilon', type=float)
parser.add_argument('--layernum', type=int, default=12)

parser.add_argument('--mode-rewrite', type=str, default='editing')
parser.add_argument('--nconcept', type=int)
parser.add_argument('--ntrain', type=int)
parser.add_argument('--nsteps', type=int, default=40000)
parser.add_argument('--nsteps-proj', type=int)
parser.add_argument('--use-mask', type=bool)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--mode-concept', type=str)
parser.add_argument('--restrict-rank', type=bool)
parser.add_argument('--rank', type=int)

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--save-checkpoint', type=bool, default=False)
parser.add_argument('--cache-dir', type=str)
parser.add_argument('--cov-dir', type=str)
parser.add_argument('--anno-filename', type=str)
parser.add_argument('--anno-min-classes', type=int)
parser.add_argument('--anno-keys', type=list)
parser.add_argument('--conf-thresh', type=float)
parser.add_argument('--pixel-thresh', type=int)

def main(args):
    
    args = grid.init_args(args)
    
    print(args)
   
    if args.styles == []:
        args.styles = os.listdir(os.path.join(args.style_dir, f'{args.dataset_name}/all_styles'))
        
    print(args.styles)
        
    # Load dataset and model
    print("----Loading dataset and model----")
    ret = lh.get_default_paths(args.dataset_name, args.epsilon, arch=args.arch)
    data_path, model_path, model_class, arch, class_dict = ret
    class_dict = {k: v.split(',')[0] for k, v in class_dict.items()}

    _, dataset = lh.get_interface_data(args.dataset_name, data_path, only_std=True)
    ret = coh.reload_classifier(model_path, model_class, arch, args.dataset_name, args.layernum) 
    
    model, context_model, target_model = ret[:3]
    preprocessing_transform = None
    if args.arch.startswith('clip'):
        dataset.transform_test = ret[-1]
        preprocessing_transform = ret[-1]
    _, val_loader = dataset.make_loaders(workers=args.num_workers, 
                                         batch_size=args.batch_size, 
                                         shuffle_val=False)
  
    print(f"----Loaded {arch}, {args.dataset_name}, {args.layernum}, {args.epsilon}, {model_path}----")
    
    # Pre-edit model accuracy
    cache_file = f'{args.cache_dir}/{arch}_{args.dataset_name}_{args.epsilon}.pt'
    _, _, acc_pre = gh.eval_accuracy(model, val_loader, batch_size=args.batch_size, cache_file=cache_file)
    ZM_k = None
    if args.mode_rewrite == 'editing':
        cov_name = f"{args.dataset_name}_{arch}_{args.epsilon}_layer{args.layernum}_imgs{len(val_loader.dataset.targets)}"
        _, ZM_k = coh.get_cov_matrix(val_loader, context_model, 
                                batch_size=2000, 
                                key_method=args.mode_concept,
                                caching_dir=os.path.join(args.cov_dir, cov_name))
                
    # Start gridding
    print(f"---Load all concept segmentations----")
    cache_file = f'{args.cache_dir}/concepts_{args.concepts}_{args.dataset_name}.npy'
    all_concepts = grid.load_concepts(args, val_loader, cache_file=cache_file)
    log_keys = {'train': ['imgs', 'manip_imgs'],
                'test': ['imgs', 'manip_imgs_same', 'manip_imgs_diff']}
      
    # Setup logging per style and concept
    args.out_dir = os.path.join(args.out_dir, f'{args.dataset_name}_{args.arch}_{args.epsilon}_{args.mode_rewrite}')
    if not os.path.exists(args.out_dir): 
        os.makedirs(args.out_dir, exist_ok=True)
        
    rng = np.random.RandomState(args.random_seed)
        
    for sidx, style_name in enumerate(args.styles):
        print(f"Style: {style_name} | {sidx + 1}/{len(args.styles)}")
        
        for cidx, concept_name in enumerate(all_concepts['concepts']):
            print(f"Style: {concept_name} | {cidx + 1}/{len(all_concepts['concepts'])}")
            
            store_name = f'{args.expt_name}_{style_name}_{concept_name}'
            
            if os.path.exists(os.path.join(args.out_dir, store_name, 'store.h5')):
                print("Trying to resume...")
                try:
                    store_test = Store(args.out_dir, store_name, new=False, mode='r')
                    if all([k in store_test.tables.keys() for k in ['metadata', 'data_info', 'results']]):
                        print(f"{concept_name}-{style_name} pair done before")
                        continue
                    else:
                        store_test.close()
                        print("Failed to resume...")
                except:
                    print("Failed to even open")
                    print("DELETING")
                    os.remove(os.path.join(args.out_dir, store_name, 'store.h5'))

                
            
            store = grid.setup_store_with_metadata(args, store_name=store_name)

            store.add_table('data_info', grid.DATA_INFO_SCHEMA)
            store.add_table('results', grid.RESULTS_SCHEMA)
        
            # Get train info
            cache_file = f'{args.cache_dir}/concept_{args.dataset_name}_{args.concepts}_{concept_name}.pt'
            concept_info = grid.load_single_concept(args, all_concepts, concept_name, 
                                                    val_loader,
                                                    preprocess=preprocessing_transform,
                                                    cache_file=cache_file)
            
            data_dict, data_info_dict = grid.obtain_train_test_splits(args, concept_info, 
                                                                      class_dict, style_name, 
                                                                      preprocess=preprocessing_transform,
                                                                      rng=rng)
            data_info_dict.update({'style_name': style_name, 'concept_name': concept_name})
            del concept_info
            
            print("Loading model")
            # Reload model
            ret = coh.reload_classifier(model_path, model_class, arch, args.dataset_name, args.layernum) 
            _, context_model, target_model = ret[:3]
    
            print("Getting predictions pre-edit...")
            # Pre-modification eval
            result_dict = {}
            for m in ['train', 'test']:
                for k2 in log_keys[m]: 
                    result_dict[f'{m}_pre_{k2}'] = gh.get_preds(context_model, data_dict[f'{m}_data'][k2],
                                                               BS=args.batch_size).numpy()
                    
            print("Editing...")
            # Rewrite model
            context_model = grid.edit_model(args, data_dict, context_model, target_model=target_model, ZM_k=ZM_k)
           
            print("Post-edit eval...")
            # Post-modification eval
            for m in ['train', 'test']:
                for k2 in log_keys[m]: 
                    result_dict[f'{m}_post_{k2}'] = gh.get_preds(context_model, data_dict[f'{m}_data'][k2],
                                                                BS=args.batch_size).numpy()
            
            _, _, acc_post = gh.eval_accuracy(context_model, val_loader, batch_size=args.batch_size, cache_file=None)
            result_dict.update({'pre_acc': acc_pre, 'post_acc': acc_post})

            print("----Logging----")
            store['data_info'].append_row(data_info_dict)
            store['results'].append_row(result_dict)
            store.close()
            grid.save_checkpoint(args, context_model)
            del result_dict
        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

