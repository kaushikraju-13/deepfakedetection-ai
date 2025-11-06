// Wait for the HTML document to be fully loaded before running the script
document.addEventListener('DOMContentLoaded', () => {

    // --- 1. Get references to all HTML elements ---
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const uploadLabel = document.getElementById('upload-label');
    const fileName = document.getElementById('file-name');
    const detectButton = document.getElementById('detect-button');
    const spinner = document.getElementById('spinner');
    const resultsArea = document.getElementById('results-area');

    // Store the selected file
    let selectedFile = null;

    // --- 2. Define API endpoint ---
    // This must match the URL and port of your Python backend
    const API_URL = 'http://127.0.0.1:8000/detect';

    // --- 3. Add Event Listeners ---

    // Handle file selection from the 'browse' button
    fileInput.addEventListener('change', (e) => {
        handleFile(e.target.files[0]);
    });

    // Handle drag-and-drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadLabel.addEventListener(eventName, preventDefaults, false);
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadLabel.addEventListener(eventName, () => uploadLabel.style.backgroundColor = '#f5f8ff', false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        uploadLabel.addEventListener(eventName, () => uploadLabel.style.backgroundColor = '#fdfdff', false);
    });
    uploadLabel.addEventListener('drop', (e) => {
        handleFile(e.dataTransfer.files[0]);
    });

    // Handle the 'Detect' button click
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // Stop the form from submitting normally
        if (!selectedFile) return;

        await sendFileToBackend();
    });

    // --- 4. Define Core Functions ---

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    /**
     * Updates the UI when a file is selected.
     * @param {File} file - The file selected by the user.
     */
    function handleFile(file) {
        if (file) {
            selectedFile = file;
            fileName.textContent = file.name;
            detectButton.disabled = false; // Enable the 'Detect' button

            // Show a video or image icon based on file type
            if (file.type.startsWith('video/')) {
                uploadLabel.querySelector('.icon').textContent = '🎬';
            } else {
                uploadLabel.querySelector('.icon').textContent = '🖼️';
            }
        }
    }

    /**
     * Sends the file to the backend API and handles the response.
     */
    async function sendFileToBackend() {
        // Show loading spinner, hide old results
        spinner.style.display = 'block';
        resultsArea.style.display = 'none';
        resultsArea.innerHTML = '';
        detectButton.disabled = true;

        // Create a FormData object to send the file
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            // Send the file to the Python backend
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            displayResult(result);

        } catch (error) {
            console.error('Error:', error);
            displayResult({ success: false, error: 'Failed to connect to the server. Is it running?' });
        } finally {
            // Hide spinner and re-enable button
            spinner.style.display = 'none';
            detectButton.disabled = false;
        }
    }

    /**
     * Displays the prediction result in the results area.
     * @param {object} result - The JSON object from the backend.
     */
    function displayResult(result) {
        resultsArea.style.display = 'block';
        let cardHTML = '';

        if (result.success) {
            const label = result.label; // "REAL" or "FAKE"
            const confidence = (result.confidence * 100).toFixed(2);
            const isFake = label === 'FAKE';
            const cardClass = isFake ? 'fake' : 'real';
            const labelClass = isFake ? 'fake' : 'real';

            cardHTML = `
                <div class="result-card ${cardClass}">
                    <div class="result-label ${labelClass}">${label}</div>
                    <div class="result-confidence">
                        <strong>Confidence:</strong> ${confidence}%
                    </div>
                </div>
            `;
        } else {
            // Display an error message
            cardHTML = `
                <div class="result-card fake">
                    <div class="result-label fake">Error</div>
                    <div class="result-error">
                        ${result.error || 'An unknown error occurred.'}
                    </div>
                </div>
            `;
        }
        resultsArea.innerHTML = cardHTML;
    }
});
