from diffusers import StableDiffusionInpaintPipeline
import torch
import gradio as gr
import numpy as np
import PIL
from PIL import Image

# Load the Stable Diffusion Inpainting model
model_id = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe.to("cpu")

def is_valid_image(image) -> bool:
    r"""
    Checks if the input is a valid image.

    A valid image can be:
    - A `PIL.Image.Image`.
    - A 2D or 3D `np.ndarray` or `torch.Tensor` (grayscale or color image).

    Args:
        image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`):
            The image to validate. It can be a PIL image, a NumPy array, or a torch tensor.

    Returns:
        `bool`:
            `True` if the input is a valid image, `False` otherwise.
    """
    return isinstance(image, PIL.Image.Image) or isinstance(image, (np.ndarray, torch.Tensor)) and image.ndim in (2, 3)

# Define the function to change the background
def change_background(prompt, image, mask_image):
    print("########### change_background function ###################")
    
    # Check if the image is already a PIL Image
    if isinstance(image, Image.Image):
        print("Image is already a PIL Image.")
    elif isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image, assuming it's in a correct format
        image = Image.fromarray(image.astype(np.uint8))
        print("Converted numpy array to PIL Image.")
    elif isinstance(image, torch.Tensor):
        # Make sure the tensor is on CPU and convert to PIL Image
        if image.is_cuda:
            image = image.cpu()
        image = Image.fromarray(image.numpy().astype(np.uint8))
        print("Converted torch Tensor to PIL Image.")
    else:
        raise ValueError("Unsupported image format. Please provide a PIL Image, numpy array, or torch Tensor.")

    # Validate the image format before passing to the model
    if not isinstance(image, Image.Image):
        raise ValueError("Image processing requires a PIL Image format after conversion.")
    
    # Check if the image is already a PIL Image
    if isinstance(mask_image, Image.Image):
        print("Image is already a PIL Image.")
    elif isinstance(mask_image, np.ndarray):
        # Convert numpy array to PIL Image, assuming it's in a correct format
        mask_image = Image.fromarray(mask_image.astype(np.uint8))
        print("Converted numpy array to PIL Image.")
    elif isinstance(mask_image, torch.Tensor):
        # Make sure the tensor is on CPU and convert to PIL Image
        if mask_image.is_cuda:
            mask_image = mask_image.cpu()
        mask_image = Image.fromarray(mask_image.numpy().astype(np.uint8))
        print("Converted torch Tensor to PIL Image.")
    else:
        raise ValueError("Unsupported image format. Please provide a PIL Image, numpy array, or torch Tensor.")

    # Validate the image format before passing to the model
    if not isinstance(mask_image, Image.Image):
        raise ValueError("Mask Image processing requires a PIL Image format after conversion.")
    
    # Log image type before passing to pipeline
    print(f"Final image type before pipe call: {type(mask_image)}")
    
    # Process the image with the Stable Diffusion Inpainting model
    result = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]

    return result

# Create a Gradio interface
iface = gr.Interface(
    fn=change_background,
    inputs=["text", "image", "image"],
    outputs="image",
    title="Change Image Background"
)

iface.launch()


# VERSION 4
# from diffusers import StableDiffusionImg2ImgPipeline
# import torch
# import gradio as gr
# from PIL import Image
# import numpy as np
# from torchvision import transforms
# import io

# # Load the Stable Diffusion Image-to-Image model
# model_id = "CompVis/stable-diffusion-v1-4"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
# pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe.to("cpu")

# # Define the function to change the background
# def change_background(prompt, image):
#     # Ensure the image is in the correct format
#     if isinstance(image, np.ndarray):
#         print("Image is a numpy array")
#         image = Image.fromarray(image)
#     elif isinstance(image, str):  # If the image is a file path
#         print("Image is a file path")
#         image = Image.open(image)
#     elif isinstance(image, bytes):  # If the image is in bytes
#         print("Image is in bytes")
#         image = Image.open(io.BytesIO(image))
#     elif isinstance(image, Image.Image):
#         print("Image is a PIL Image")
#     else:
#         print("Unknown image format")
#         raise ValueError("Unsupported image format")

#     # Define the necessary transform
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
    
#     # Convert the PIL image to a tensor and ensure it's 3D
#     image_tensor = transform(image).unsqueeze(0).to(pipe.device)  # Add batch dimension and move to device
    
#     # Ensure the tensor is 4D
#     if image_tensor.ndim == 3:
#         image_tensor = image_tensor.unsqueeze(0)
    
#     # Generate the modified image
#     result = pipe(prompt=prompt, init_image=image_tensor).images[0]
#     return result

# # Create a Gradio interface
# iface = gr.Interface(
#     fn=change_background,
#     inputs=["text", "image"],
#     outputs="image",
#     title="Change Image Background"
# )

# iface.launch()



#  VERSION 3
# from diffusers import StableDiffusionImg2ImgPipeline
# import torch
# import gradio as gr
# from PIL import Image
# import numpy as np
# from torchvision import transforms
# import io

# # Load the Stable Diffusion Image-to-Image model
# model_id = "CompVis/stable-diffusion-v1-4"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
# pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe.to("cpu")

# # Define the function to change the background
# def change_background(prompt, image):
#     # Define the necessary transform
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     if isinstance(image, np.ndarray):
#         image = Image.fromarray(image)
#     elif isinstance(image, str):  # If the image is a file path
#         image = Image.open(image)
#     elif isinstance(image, bytes):  # If the image is in bytes
#         image = Image.open(io.BytesIO(image))
#     elif not isinstance(image, Image.Image):
#         raise ValueError("Unsupported image format")

#     # Convert the PIL image to a tensor
#     image_tensor = transform(image).unsqueeze(0).to(pipe.device)  # Add batch dimension and move to device
    
#     # Generate the modified image
#     result = pipe(prompt=prompt, init_image=image_tensor).images[0]
#     return result

# # Create a Gradio interface
# iface = gr.Interface(
#     fn=change_background,
#     inputs=["text", "image"],
#     outputs="image",
#     title="Change Image Background"
# )

# iface.launch()



# VERSION 2
# from diffusers import StableDiffusionImg2ImgPipeline
# import torch
# import gradio as gr
# from PIL import Image
# import numpy as np
# import io

# # Load the Stable Diffusion Image-to-Image model, ensuring CPU compatibility with float32
# model_id = "CompVis/stable-diffusion-v1-4"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
# pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe.to("cpu")

# # Define the function to change the background
# def change_background(prompt, image):
#     # Convert numpy array to PIL.Image.Image if necessary
#     if isinstance(image, np.ndarray):
#         image = Image.fromarray(image)
#     elif isinstance(image, bytes):  # If the image is in bytes
#         image = Image.open(io.BytesIO(image))
    
#     # Ensure image is in RGB mode and resized
#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     image = image.resize((512, 512))  # Resize to standard model input size
    
#     # Pass the PIL image to the pipeline
#     try:
#         result = pipe(prompt=prompt, init_image=image, strength=0.75).images[0]
#     except Exception as e:
#         print(f"Error during image generation: {e}")
#         raise e
    
#     return result

# # Create a Gradio interface
# iface = gr.Interface(
#     fn=change_background,
#     inputs=["text", "image"],
#     outputs="image",
#     title="Change Image Background"
# )

# iface.launch()
