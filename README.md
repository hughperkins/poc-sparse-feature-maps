# poc-sparse-feature-maps
POC for using sparse feature maps

## Vision

It could be pleasing to have the final layer of some future cnn have one feature map for each type of object, eg bicycles, grand pianos and stuff.  If you put a '1' in one of these feature maps, and then back-propagated onto white noise, it could be cool if that somehow generated eg a grand piano at that position in the image.

For this to happen, we would need maybe ten thousand or a million feature maps at the output.  Most of these would be all zeros: they will be sparse.  Since there are so many of them, ideally we need some kind of sparse representation

There are two parts to achieving this:
- theoretical: how to enforce sparseness?
- engineering: how to store sparse tensors, implement sparse library etc

## Engineering / representation

### Representation

Multiple sparse representations exist. All have their own good and bad points.  Currently, I am considering a representation where each plane, in the weights and the activations, is stored densely, but not all planes are stored, some are considered to be entirely zero'd out.  This has advantages:
- facilitates implementation
- can continue to use standard GPU implementatinos for convolution etc

On the downside, it's quite non-standard, and might not achieve the same sparsity, in terms of actual storage used, compared to some more standard representation.  I'm a bit concerned too that it will have all of the implementation issues that 0-norms have, ie non-differentiability, exponentially combinatorial complexity etc :-(

### Implementation

To run poc:
```
luarocks make rocks/sparseplanar-scm-1.rockspec  && th test2.lua
```

Pre-requisites:
- torch must be installed
- torch must be activated (ie `source ~/torch/install/bin/torch-activate`)

## Theory / model

Two things need to be made sparse:
- the activations
- the weights

The weights must be sparse too, otherwise they will be massive (I guess?), or at least, propagation will need a lot of calculations.

For making the activations sparse, we can use KL divergence, eg http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity.

*However this doesn't address how to make the weights sparse*.  Also, ideally we'd enforce sparseness on a per-plane basis, though, this might encourage the difficulties associated with 0-norms, combinatorial complexity etc.

