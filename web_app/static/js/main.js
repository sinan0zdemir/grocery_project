// Global filter function for metric cards
window.filterIssues = function(filterType) {
    const items = document.querySelectorAll('.issue-item');
    items.forEach(item => {
        if (filterType === 'all') {
            item.style.display = 'flex';
        } else {
            if (item.classList.contains(filterType)) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        }
    });
};

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewContainer = document.getElementById('preview-container');
    const uploadContent = document.querySelector('.upload-content');
    const imagePreview = document.getElementById('image-preview');
    const removeImgBtn = document.getElementById('remove-img-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');
    
    // Result Elements
    const scoreBadge = document.getElementById('score-badge');
    const totalItems = document.getElementById('total-items');
    const missingItems = document.getElementById('missing-items');
    const misplacedItems = document.getElementById('misplaced-items');
    const issueList = document.getElementById('issue-list');

    let currentFile = null;

    // --- File Drag & Drop Handlers ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFiles(files[0]);
        }
    }

    // --- Click Handlers ---
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) handleFiles(this.files[0]);
    });

    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        analyzeBtn.disabled = true;
        fileInput.value = '';
    });

    function handleFiles(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (JPG, PNG).');
            return;
        }
        
        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // --- API Integration ---
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Show loading state
        loadingOverlay.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('file', currentFile);

            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('API request failed');

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
        } finally {
            loadingOverlay.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            // Scroll to results smoothly
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });

    function displayResults(data) {
        scoreBadge.textContent = `${data.compliance_score}% Match`;
        
        // Color badge based on score
        if (data.compliance_score > 90) {
            scoreBadge.style.color = 'var(--success)';
            scoreBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
            scoreBadge.style.background = 'rgba(16, 185, 129, 0.2)';
        } else if (data.compliance_score > 70) {
            scoreBadge.style.color = 'var(--warning)';
            scoreBadge.style.borderColor = 'rgba(245, 158, 11, 0.3)';
            scoreBadge.style.background = 'rgba(245, 158, 11, 0.2)';
        } else {
            scoreBadge.style.color = 'var(--danger)';
            scoreBadge.style.borderColor = 'rgba(239, 68, 68, 0.3)';
            scoreBadge.style.background = 'rgba(239, 68, 68, 0.2)';
        }

        totalItems.textContent = data.total_items;
        missingItems.textContent = data.missing_items.length;
        misplacedItems.textContent = data.misplaced_items.length;

        // Render Issue List
        issueList.innerHTML = '';
        
        if (data.missing_items.length === 0 && data.misplaced_items.length === 0 && data.unexpected_items.length === 0) { // Added unexpected_items to condition
            issueList.innerHTML = `<li class="empty-state" style="border: 1px solid var(--success); color: var(--success);">
                <i data-lucide="check-circle"></i> Perfect Planogram Match!
            </li>`;
            lucide.createIcons();
            return;
        }

        // Add Missing
        if (data.missing_items) {
            data.missing_items.forEach(item => {
                const li = document.createElement('li');
                li.className = 'issue-item missing';
                li.innerHTML = `
                    <div>
                        <strong>Missing Item</strong>
                        <div style="font-size: 0.875rem; color: var(--text-secondary);">${item.label}</div>
                    </div>
                    <div style="text-align: right; color: var(--danger); font-size: 0.875rem;">
                        Expected: Shelf ${item.expected_shelf}<br>
                        Position: ${item.expected_position}
                    </div>
                `;
                issueList.appendChild(li);
            });
        }
        
        // Add Misplaced
        if (data.misplaced_items) {
            data.misplaced_items.forEach(item => {
                const li = document.createElement('li');
                li.className = 'issue-item misplaced';
                li.innerHTML = `
                    <div>
                        <strong>Misplaced Item</strong>
                        <div style="font-size: 0.875rem; color: var(--text-secondary);">Detected: ${item.detected_label}</div>
                    </div>
                    <div style="text-align: right; color: var(--warning); font-size: 0.875rem;">
                        Expected: Shelf ${item.expected_shelf} (${item.expected_label})<br>
                        Detected: Shelf ${item.detected_shelf}
                    </div>
                `;
                issueList.appendChild(li);
            });
        }

        // Add Unexpected
        if (data.unexpected_items) {
            data.unexpected_items.forEach(item => {
                const li = document.createElement('li');
                li.className = 'issue-item unexpected';
                li.innerHTML = `
                    <div>
                        <strong>Unexpected Item</strong>
                        <div style="font-size: 0.875rem; color: var(--text-secondary);">${item.label}</div>
                    </div>
                    <div style="text-align: right; color: var(--danger); font-size: 0.875rem;">
                        Detected: Shelf ${item.detected_shelf}<br>
                        Position: ${item.detected_position}
                    </div>
                `;
                issueList.appendChild(li);
            });
        }

        // Add Correct
        if (data.correct_items) {
            data.correct_items.forEach(item => {
                const li = document.createElement('li');
                li.className = 'issue-item correct';
                li.innerHTML = `
                    <div>
                        <strong>Correct Item</strong>
                        <div style="font-size: 0.875rem; color: var(--text-secondary);">${item.label}</div>
                    </div>
                    <div style="text-align: right; color: var(--success); font-size: 0.875rem;">
                        Shelf ${item.shelf}<br>
                        Position: ${item.position}
                    </div>
                `;
                issueList.appendChild(li);
            });
        }

        // Add Gap Detections (Physical Empty Spaces)
        if (data.gap_detections && data.gap_detections.length > 0) {
            data.gap_detections.forEach(item => {
                const li = document.createElement('li');
                li.className = 'issue-item missing';
                li.innerHTML = `
                    <div>
                        <strong>🕳️ Boş Alan Tespit Edildi</strong>
                        <div style="font-size: 0.875rem; color: var(--text-secondary);">${item.label}</div>
                    </div>
                    <div style="text-align: right; color: #a855f7; font-size: 0.875rem;">
                        Raf ${item.expected_shelf}<br>
                        Fiziksel Boşluk
                    </div>
                `;
                issueList.appendChild(li);
            });
        }
    }
});
