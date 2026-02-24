let pieChartInstance = null;
let lineChartInstance = null;

chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {

    const url = tabs[0]?.url;

    const totalElement = document.getElementById("total");

    if (!url || !url.includes("youtube.com") && !url.includes("youtu.be")) {
        totalElement.innerText = "Please open a valid YouTube video.";
        return;
    }

    totalElement.innerText = "Analyzing comments...";

    fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.detail || "Server Error");
            });
        }
        return response.json();
    })
    .then(data => {

        totalElement.innerText = "Total Comments: " + data.total_comments;

        // Destroy old charts if they exist
        if (pieChartInstance) pieChartInstance.destroy();
        if (lineChartInstance) lineChartInstance.destroy();

        // PIE CHART
        const pieCtx = document.getElementById("pieChart").getContext("2d");

        pieChartInstance = new Chart(pieCtx, {
            type: "pie",
            data: {
                labels: ["Positive", "Neutral", "Negative"],
                datasets: [{
                    data: [
                        data.sentiment_distribution.positive,
                        data.sentiment_distribution.neutral,
                        data.sentiment_distribution.negative
                    ],
                    backgroundColor: ["#4CAF50", "#FFC107", "#F44336"]
                }]
            },
            options: {
                plugins: {
                    legend: {
                        labels: { color: "#ffffff" }
                    }
                }
            }
        });

        // MONTHLY LINE CHART
        const months = Object.keys(data.monthly_counts);

        if (months.length === 0) {
            return;
        }

        const positives = months.map(m => data.monthly_counts[m].positive || 0);
        const neutrals = months.map(m => data.monthly_counts[m].neutral || 0);
        const negatives = months.map(m => data.monthly_counts[m].negative || 0);

        const lineCtx = document.getElementById("lineChart").getContext("2d");

        lineChartInstance = new Chart(lineCtx, {
            type: "line",
            data: {
                labels: months,
                datasets: [
                    {
                        label: "Positive",
                        data: positives,
                        borderColor: "#4CAF50",
                        tension: 0.3
                    },
                    {
                        label: "Neutral",
                        data: neutrals,
                        borderColor: "#FFC107",
                        tension: 0.3
                    },
                    {
                        label: "Negative",
                        data: negatives,
                        borderColor: "#F44336",
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        ticks: { color: "#cccccc" }
                    },
                    y: {
                        ticks: { color: "#cccccc" }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: "#ffffff" }
                    }
                }
            }
        });

    })
    .catch(error => {
        totalElement.innerText = "Error: " + error.message;
    });

});