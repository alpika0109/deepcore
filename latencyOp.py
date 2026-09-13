device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# During inference, move your tensors to the device
def preprocess_and_predict(frame1, frame2):
    tensor1 = preprocess_frame(frame1).unsqueeze(0).to(device)  # add batch dimension
    tensor2 = preprocess_frame(frame2).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(tensor1, tensor2)
        prediction = torch.argmax(output, dim=1).item()
    return prediction
