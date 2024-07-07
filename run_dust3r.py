from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.demo import get_3D_model_from_scene
import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    )
    parser.add_argument("--scene_graph", type=str, default="complete")
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=["croco/assets/Chateau1.png", "croco/assets/Chateau2.png"],
    )
    parser.add_argument("--outdir", type=str, default="io/output/")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--schedule", type=str, default="cosine")

    args = parser.parse_args()

    device = args.device
    batch_size = 1
    schedule = args.schedule
    lr = args.lr
    niter = args.niter
    model_name = args.model_name
    outdir = args.outdir
    silent = args.silent

    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    image_paths = args.images
    if len(args.images) == 1:
        image_paths = glob.glob(args.images[0])
        image_paths = sorted(image_paths)[:40]
    images = load_images(image_paths, size=512)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    # view1, pred1 = output['view1'], output['pred1']
    # view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(
        output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer
    )
    loss = scene.compute_global_alignment(
        init="mst", niter=niter, schedule=schedule, lr=lr
    )

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # Create glb file
    glb_model = get_3D_model_from_scene(
        outdir,
        silent,
        scene,
        min_conf_thr=3,
        as_pointcloud=False,
        mask_sky=False,
        clean_depth=False,
        transparent_cams=False,
        cam_size=0.03,
    )
