let loadingInterval = null;

function startLoadingAnimation() {
    const loading = document.getElementById("loading");

    const messages = [
        "Scraping listing...",
        "Extracting property details...",
        "Hafillama is waking up...",
        "Analyzing price patterns...",
        "Fetching the deals to the Fridge...",
        "Comparing with best deals...",
        "Hafillama is Hafeez?...",
        "Calculating deal score...",
        "Preparing final verdict...",
        "Eating Doner...",
        "Hafilama is Almost there...",
        "Almost there...",
    ];

    let index = 0;
    let dots = "";

    loading.classList.remove("hidden");
    loading.innerText = messages[index];

    loadingInterval = setInterval(() => {
        dots = dots.length < 3 ? dots + "." : "";

        loading.innerText = messages[index] + dots;

        index = (index + 1) % messages.length;
    }, 1200);
}


function stopLoadingAnimation() {
    const loading = document.getElementById("loading");

    if (loadingInterval) {
        clearInterval(loadingInterval);
        loadingInterval = null;
    }

    loading.classList.add("hidden");
    loading.innerText = "";
}


async function analyzeDeal() {
    const url = document.getElementById("urlInput").value.trim();
    const resultCard = document.getElementById("resultCard");

    if (!url) {
        alert("Please enter a listing URL");
        return;
    }

    startLoadingAnimation();
    resultCard.classList.add("hidden");

    try {
        const response = await fetch("/deals/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ url: url })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Something went wrong");
        }

        document.getElementById("prediction").innerText = data.prediction;
        document.getElementById("confidence").innerText = data.confidence + "%";
        document.getElementById("pricePerSqm").innerText = "€" + data.price_per_sqm;
        document.getElementById("valueScore").innerText = data.value_score;
        document.getElementById("bestDealsFound").innerText = data.best_deals_found;

        document.getElementById("bestAvg").innerText =
            data.best_deals_avg_price_per_sqm
                ? "€" + data.best_deals_avg_price_per_sqm
                : "N/A";

        document.getElementById("finalExplanation").innerText =
            data.final_explanation || "No explanation available.";

        resultCard.classList.remove("hidden");

    } catch (error) {
        alert(error.message);
    } finally {
        stopLoadingAnimation();
    }
}


document.getElementById("urlInput").addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
        analyzeDeal();
    }
});