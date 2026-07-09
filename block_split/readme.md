# CNN/GNN Urban LCZ–LST Block-Split Experiments

This folder contains block-split versions of the CNN and GNN experiments for urban LCZ-based LST prediction.

The purpose is to test spatial generalization. The earlier random pixel split is useful as a baseline, but nearby pixels often share similar LCZ context, climate background, and LST values. In a random split, train and test pixels may be very close to each other. -> so that the performance may be too good to be challenged by reviewers.

In the block split, valid urban pixels are first assigned to spatial blocks based on their raster row and column. Whole blocks are then assigned to train, validation, or test sets. -> the model is evaluated on spatial areas that were held out during training.

**Pixel split** asks whether the model can predict missing pixels among nearby known pixels. 
**Block split** asks whether the model can predict a held-out spatial area.

No coordinate predictors as well.

For the CNN, each target pixel is predicted from a local patch of LCZ-derived feature maps. The default patch size is 9 $\times$ 9 pixels. The model is trained on pixels from training blocks, tuned on validation blocks, and evaluated on test blocks.

For the GNN, valid urban pixels are represented as graph nodes. Each node has LCZ-derived landscape features. Local graph edges are constructed using k-nearest neighbors, usually k = 8 (as default in the very first version). For the cleanest validation, the GNN graph should be built separately for train, validation, and test nodes, or edges crossing train/validation/test groups should be removed. -> it avoids message passing between training and held-out nodes/pixels.

The block size is defined in pixels. For 2 km data, block_size = 32 means about 64 km $\times$ 64 km blocks. For 500 m data, block_size = 32 means about 16 km $\times$ 16 km blocks (**feel free to adjust this value**). Smaller blocks are easier but closer to local interpolation. Larger blocks are stricter but may leave fewer samples.

Please record for each experiment: target variable, resolution, spatial domain, input predictors, split type, block size, train/validation/test sample counts, CNN patch size, GNN k value, and test R square/RMSE/MAE (or whatever metrics we used).

Recommended order: keep the current no-coordinate random split as a baseline (Negar, you already have it), then run the no-coordinate block split as the main robustness test (in this folder). If we want to claim transfer across cities, we need to also add a city-holdout or megapolitan-area-holdout test.

High random-split R square mainly shows interpolation skill. High block-split R square can give stronger evidence of spatial generalization.
