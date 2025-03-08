let selectedModel = '';  // To store selected model

function selectModel(model) {
    selectedModel = model;  // Update selected model
    console.log("Model selected:", selectedModel);
}

async function predict() {
    if (!selectedModel) {
        alert("Please select a model first!");
        return;
    }

    const features = document.getElementById("features").value.split(",").map(Number);

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: selectedModel, features })
    });

    const result = await response.json();
    document.getElementById("result").innerText = "Prediction: " + JSON.stringify(result);
}
