import torch
import random
import torchvision
import torchvision.transforms as T
import numpy as np
from .image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from scipy.ndimage.filters import gaussian_filter1d


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()
    model = torchvision.models.squeezenet1_1(pretrained=True)

    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # FORWARD pass: Get scores, then filter out only those entries
    # which are for the correct class (as this is Softmax loss).
    scores = model(X)
    correct_scores = scores.gather(1, y.view(-1, 1)).squeeze()

    # BACKWARD pass: We can call torch.Tensor.backward() on arbitrary losses -
    # this is the auto-differentiation engine doing its job.

    # Since correct_scores.requires_grad = true, we must pass the gradient arg.
    #
    # The gradient arg is: "Gradient w.r.t. the tensor [and is] a tensor of matching
    # type and location, [containing] the grad of the differentiated function w.r.t. self."
    #
    # Simple math - we know dX/dX = [1] (arr of ones in shape of X), so this is easy:
    correct_scores.backward(torch.ones(y.shape[0]))

    # torch.Tensor.backward() stores its outputs by updating the tensor's .grad attribute.
    dImage_dModelWeights = X.grad

    # Per the paper, we then take the absolute value and keep the max val across channels:
    saliency, _ = torch.max(np.absolute(dImage_dModelWeights), dim=1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image, and make it require gradient
    X_fooling = X.clone()
    X_fooling = X_fooling.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. You should perform gradient ascent on the score of the #
    # target class, stopping when the model is fooled.                           #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    iters = 100

    # for i in range(iters):
    #     scores = model(X_fooling)
    #     target_score = scores[:, target_y]
    #
    #     if torch.argmax(scores) == target_y:

    scores = model(X_fooling)
    i = 0
    # This loop should generally be able to fool in <100 iterations.
    while torch.argmax(scores) != target_y and 1 <= 100:
        scores = model(X_fooling)
        target_score = scores[:, target_y]

        target_score.backward()  # Output is written to X_fooling.grad
        dx = learning_rate * X_fooling.grad / np.linalg.norm(X_fooling.grad)

        with torch.no_grad():
            X_fooling += learning_rate * dx

        i += 1

        if torch.argmax(scores) == target_y:
            print("It worked!")

    if i == 100:
        print("Nothing after 100 iters boss D:")

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling


def class_visualization_update_step(img, model, target_y, l2_reg, learning_rate):
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    ########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Note: This isn't on the 'raw' score, but on the l2-regularized score
    # this is -- almost -- like a loss function
    #
    # Per the suggested paper's notes (section 2), we are l2-regularizing
    # the scores _before_ the posterior softmax.
    #
    # Then, we're solving a simple optimization problem - find argmax S_c(I).
    #
    # By formulating this as a maximization on S_c (ignoring all other classes)
    # we do something mathematically similar to maximizing on the softmax (loss) f'n,
    # but this little 'trick' offers more visually appealing results.
    scores = model(img)
    target_score = scores[:, target_y] - l2_reg * torch.norm(img) ** 2
    # target_score = scores[:, target_y] - l2_reg * torch.sum(img * img)
    target_score.backward()

    # no return value, so we edit img in-place.
    with torch.no_grad():
        img += learning_rate * img.grad
    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################


def preprocess(img, size=224):
    transform = T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=SQUEEZENET_MEAN.tolist(), std=SQUEEZENET_STD.tolist()),
            T.Lambda(lambda x: x[None]),
        ]
    )
    return transform(img)


def deprocess(img, should_rescale=True):
    transform = T.Compose(
        [
            T.Lambda(lambda x: x[0]),
            T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
            T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
            T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
            T.ToPILImage(),
        ]
    )
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X


def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X
