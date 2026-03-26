// Global filter function for metric cards
window.filterIssues = function (filterType) {
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
    const setRefBtn = document.getElementById('set-ref-btn');
    const clearRefBtn = document.getElementById('clear-ref-btn');
    const refStatusText = document.getElementById('ref-status-text');

    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');

    // Result Elements
    const scoreBadge = document.getElementById('score-badge');
    const totalItems = document.getElementById('total-items');
    const missingItems = document.getElementById('missing-items');
    const misplacedItems = document.getElementById('misplaced-items');
    const issueList = document.getElementById('issue-list');

    let currentFile = null;

    // --- Reference Status ---
    async function checkReferenceStatus() {
        try {
            const res = await fetch('/api/check_reference');
            const data = await res.json();
            if (data.has_reference) {
                refStatusText.textContent = 'Current Reference: Active (Golden Image)';
                refStatusText.style.color = 'var(--success)';
                clearRefBtn.classList.remove('hidden');
            } else {
                refStatusText.textContent = 'Current Reference: None (Heuristics Only)';
                refStatusText.style.color = 'var(--text-secondary)';
                clearRefBtn.classList.add('hidden');
            }
        } catch (e) {
            console.error(e);
        }
    }
    checkReferenceStatus();

    clearRefBtn.addEventListener('click', async () => {
        if (confirm("Are you sure you want to clear the Golden Image reference?")) {
            await fetch('/api/clear_reference', { method: 'POST' });
            checkReferenceStatus();
            alert("Reference cleared.");
        }
    });

    // --- File Drag & Drop Handlers ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

    ['dragenter', 'dragover'].forEach(eventName => { dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false); });
    ['dragleave', 'drop'].forEach(eventName => { dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false); });

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) handleFiles(files[0]);
    }

    // --- Click Handlers ---
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function () {
        if (this.files.length > 0) handleFiles(this.files[0]);
    });

    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        analyzeBtn.disabled = true;
        setRefBtn.disabled = true;
        fileInput.value = '';
    });

    function handleFiles(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (JPG, PNG).');
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
            setRefBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // --- API Integration ---
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        loadingOverlay.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('file', currentFile);
            const response = await fetch('/api/analyze', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('API request failed');
            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
        } finally {
            loadingOverlay.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });

    setRefBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        loadingOverlay.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('file', currentFile);
            const response = await fetch('/api/set_reference', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('API request failed');
            const data = await response.json();

            if (data.status === 'success') {
                alert('Successfully generated and saved Golden Image Reference!');
                checkReferenceStatus();
            } else {
                alert('Setting reference failed: ' + data.message);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while setting reference.');
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    });

    function displayResults(data) {
        scoreBadge.textContent = `Analysis Complete`;
        scoreBadge.style.color = 'var(--text-primary)';
        scoreBadge.style.borderColor = 'var(--border-color)';
        scoreBadge.style.background = 'var(--surface-color)';

        totalItems.textContent = data.total_items + " Items";

        const gapCount = data.gap_detections ? data.gap_detections.length : 0;
        const missingItemCount = data.missing_items ? data.missing_items.length : 0;
        const misplacedItemCount = data.misplaced_items ? data.misplaced_items.length : 0;

        // Show only the visual yellow boxes in the counter
        missingItems.textContent = gapCount + " Items (Yellow)";
        misplacedItems.textContent = misplacedItemCount + " Items";

        // Render Issue List
        issueList.innerHTML = '';
        if (gapCount === 0 && missingItemCount === 0 && misplacedItemCount === 0) {
            issueList.innerHTML = `<li class="empty-state" style="border: 1px solid var(--success); color: var(--success);">
                <i data-lucide="check-circle"></i> Perfect Shelf Structure! No anomalies or missing items detected.
            </li>`;
            lucide.createIcons();
            return;
        }

        // Add Exact Missing Items (From Schema)
        if (data.missing_items && data.missing_items.length > 0) {
            data.missing_items.forEach(item => {
                const li = document.createElement('li');
                li.className = 'issue-item';
                li.style.borderLeft = '4px solid var(--warning)';
                li.innerHTML = `
                    <div style="width: 100%;">
                        <strong style="display:block; margin-bottom:4px;">Missing from Inventory (Not on shelf)</strong>
                        <div style="font-size: 0.875rem; color: var(--text-secondary);">
                            <span style="color: var(--warning); font-weight: 500;">Missing:</span> ${item.label}
                        </div>
                    </div>
                    <div style="text-align: right; color: var(--text-secondary); font-size: 0.875rem;">
                        Expected on Shelf ${item.expected_shelf}
                    </div>
                `;
                issueList.appendChild(li);
            });
        }

        // Add Misplaced
        if (data.misplaced_items && data.misplaced_items.length > 0) {
            data.misplaced_items.forEach(item => {
                const li = document.createElement('li');
                li.className = 'issue-item misplaced';
                const expectedText = item.expected_label || 'Unknown';
                const foundText = item.detected_label || item.label || 'Unknown Item';
                li.innerHTML = `
                    <div style="width: 100%;">
                        <strong style="display:block; margin-bottom:6px;">Misplaced Item (Wrong Shelf)</strong>
                        <div style="font-size: 0.875rem; background: rgba(0,0,0,0.02); padding: 6px; border-radius: 4px; border-left: 3px solid var(--success); margin-bottom: 4px;">
                            <span style="color: var(--success); font-weight: 600;">Expected:</span> <span style="color: var(--text-secondary);">${expectedText}</span>
                        </div>
                        <div style="font-size: 0.875rem; background: rgba(0,0,0,0.02); padding: 6px; border-radius: 4px; border-left: 3px solid var(--danger);">
                            <span style="color: var(--danger); font-weight: 600;">Found:</span> <span style="color: var(--text-primary); font-weight: 500;">${foundText}</span>
                        </div>
                        <div style="font-size: 0.75rem; color: #888; margin-top: 6px; font-style: italic;">
                            ${item.detail_msg || ''}
                        </div>
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
                        <strong style="display:block; margin-bottom:4px;">Missing Item Detected (Physical Gap)</strong>
                        <div style="font-size: 0.875rem; color: var(--text-secondary);">
                            <span style="color: var(--warning); font-weight: 500;">Space for:</span> ${item.label}
                        </div>
                    </div>
                    <div style="text-align: right; color: var(--warning); font-size: 0.875rem;">
                        Shelf ${item.expected_shelf}<br>
                        Physical Gap
                    </div>
                `;
                issueList.appendChild(li);
            });
        }

        lucide.createIcons();
    }
});
