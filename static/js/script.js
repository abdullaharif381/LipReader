document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const videoUpload = document.getElementById('video-upload');
    const fileName = document.getElementById('file-name');
    const submitBtn = document.getElementById('submit-btn');
    const loading = document.getElementById('loading');
    const resultContainer = document.getElementById('result-container');
    const transcriptionResult = document.getElementById('transcription-result');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');

    // Display the selected file name
    videoUpload.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            fileName.textContent = this.files[0].name;
            submitBtn.disabled = false;
        } else {
            fileName.textContent = '';
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Validate file
        if (!videoUpload.files || !videoUpload.files[0]) {
            showError('Please select a video file.');
            return;
        }
        
        const file = videoUpload.files[0];
        
        // Check file type
        if (!file.name.toLowerCase().endsWith('.mpg')) {
            showError('Only .mpg files are supported.');
            return;
        }
        
        // Create FormData object
        const formData = new FormData();
        formData.append('video', file);
        
        // Show loading spinner
        hideAllContainers();
        loading.classList.remove('hidden');
        
        // Send the request
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideAllContainers();
            
            if (data.success) {
                // Show transcription result
                transcriptionResult.textContent = data.transcription;
                resultContainer.classList.remove('hidden');
            } else {
                // Show error message
                showError(data.error);
            }
        })
        .catch(error => {
            hideAllContainers();
            showError('An error occurred while processing your request.');
            console.error('Error:', error);
        });
    });

    function hideAllContainers() {
        loading.classList.add('hidden');
        resultContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorContainer.classList.remove('hidden');
    }
});