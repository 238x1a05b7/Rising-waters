document.addEventListener('DOMContentLoaded', () => {
    const predictForm = document.getElementById('predictForm');
    const predictBtn = document.getElementById('predictBtn');
    const resultContainer = document.getElementById('resultContainer');
    const resultText = document.getElementById('resultText');

    predictForm.addEventListener('submit', async function(e) {
        e.preventDefault(); 

        predictBtn.innerText = "Calculating Risk...";
        predictBtn.disabled = true;
        
        if (resultContainer) {
            resultContainer.classList.add('hidden');
            // Reset background styles
            resultContainer.style.backgroundColor = '';
            resultContainer.style.borderColor = '';
            resultContainer.style.color = '';
        }

        const payloadData = {
            Temperature: document.getElementById('Temperature').value,
            Humidity: document.getElementById('Humidity').value,
            Cloud_Cover: document.getElementById('Cloud_Cover').value,
            Annual_Rainfall: document.getElementById('Annual_Rainfall').value,
            Jan_Feb: document.getElementById('Jan_Feb').value,
            Mar_May: document.getElementById('Mar_May').value,
            Jun_Sep: document.getElementById('Jun_Sep').value,
            Oct_Dec: document.getElementById('Oct_Dec').value,
            Avg_June: document.getElementById('Avg_June').value,
            Flood_Risk_Index: document.getElementById('Flood_Risk_Index').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payloadData)
            });

            const result = await response.json();

            if (result.success) {
                resultText.innerText = result.prediction;
                resultContainer.classList.remove('hidden');

                // Apply dynamic background coloring based on text output
                if (result.prediction.includes("High Chance")) {
                    resultContainer.style.backgroundColor = "#fef2f2"; // Soft Red
                    resultContainer.style.borderColor = "#fca5a5";
                    resultContainer.style.color = "#991b1b";
                } else {
                    resultContainer.style.backgroundColor = "#f0fdf4"; // Soft Green
                    resultContainer.style.borderColor = "#bbf7d0";
                    resultContainer.style.color = "#166534";
                }
            } else {
                alert("Model Execution Failure:\n" + result.error);
            }

        } catch (err) {
            console.error("API Communication Error:", err);
            alert("Failed to communicate with your Flask backend.");
        } finally {
            predictBtn.innerText = "☁️ Predict";
            predictBtn.disabled = false;
        }
    });
});