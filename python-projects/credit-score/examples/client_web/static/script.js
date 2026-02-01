document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('credit-form');
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const predictionBadge = document.getElementById('prediction-badge');
    const probFill = document.getElementById('prob-fill');
    const probText = document.getElementById('prob-text');
    const debitCard = document.getElementById('debit-card');
    const jobSlider = document.getElementById('job-slider');
    const jobHidden = document.getElementById('job-hidden');

    // Mapeo de niveles de trabajo conforme a JobEnum
    const jobLevels = [
        "unskilled and non-resident",
        "unskilled and resident",
        "skilled",
        "highly skilled"
    ];

    // Actualizar el valor oculto del Job al mover el slider
    jobSlider.addEventListener('input', (e) => {
        jobHidden.value = jobLevels[e.target.value];
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Limpiar resultados previos
        resultContainer.classList.add('hidden');

        // Activar animaciones
        submitBtn.classList.add('loading');
        debitCard.classList.add('analyzing');

        const formData = new FormData(form);

        // Construir el objeto JSON respetando los ALIAS de la API
        const payload = {
            "Age": parseInt(formData.get('Age')),
            "Sex": formData.get('Sex'),
            "Job": formData.get('Job'),
            "Housing": formData.get('Housing'),
            "Saving accounts": formData.get('Saving accounts'),
            "Checking account": formData.get('Checking account'),
            "Credit amount": parseFloat(formData.get('Credit amount')),
            "Duration": parseInt(formData.get('Duration')),
            "Purpose": formData.get('Purpose')
        };

        console.log("Enviando datos:", payload);

        try {
            // Simulamos un pequeño delay para que la animación se aprecie
            await new Promise(resolve => setTimeout(resolve, 2000));

            const response = await fetch('http://localhost:8000/credit_score_prediction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Error en la API');
            }

            const result = await response.json();
            showResult(result);

        } catch (error) {
            console.error('Error:', error);
            alert('Error al conectar con la API: ' + error.message);
        } finally {
            submitBtn.classList.remove('loading');
            debitCard.classList.remove('analyzing');
        }
    });

    function showResult(data) {
        resultContainer.classList.remove('hidden');

        const isGood = data.prediction.toLowerCase() === 'good';
        predictionBadge.textContent = isGood ? 'APROBADO' : 'RIESGO ALTO';
        predictionBadge.className = `badge ${isGood ? 'good' : 'bad'}`;

        const percentage = (data.probability * 100).toFixed(1);
        probFill.style.width = `${percentage}%`;
        probText.textContent = `${percentage}%`;

        // Scroll suave al resultado
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
});
