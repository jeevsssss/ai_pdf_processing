document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const optionsSection = document.getElementById('optionsSection');

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            // Upload the file using FormData and fetch API
            const formData = new FormData();
            formData.append('file', file);
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (response.ok) {
                    // Show the options section
                    optionsSection.style.display = 'block';
                } else {
                    console.error('Error uploading the file');
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    });

    // Add event listeners for each option button (extract keywords, find product, find negative feedback)
    // You can add these event listeners to fetch corresponding data from the backend and display the results.
});
