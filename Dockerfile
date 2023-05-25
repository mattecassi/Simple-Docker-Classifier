# Use PyTorch as the base image
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app


# Copy your code into the container
COPY ResNet18.py .


# Run the code when the container starts
CMD ["python", "ResNet18.py"]