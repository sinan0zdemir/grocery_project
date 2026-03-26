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

function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    const colors = { success: '#22c55e', error: '#ef4444', info: '#6366f1', warning: '#f59e0b' };
    toast.style.cssText = `
        background: var(--card-bg, #1e1e2e);
        border: 1px solid ${colors[type] || colors.info};
        color: var(--text-primary, #fff);
        padding: 0.75rem 1.1rem;
        border-radius: 10px;
        font-size: 0.9rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
        pointer-events: auto;
        opacity: 0;
        transform: translateX(20px);
        transition: opacity 0.25s, transform 0.25s;
        max-width: 320px;
        border-left: 4px solid ${colors[type] || colors.info};
    `;
    toast.textContent = message;
    container.appendChild(toast);
    requestAnimationFrame(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateX(0)';
    });
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

document.addEventListener('DOMContentLoaded', () => {
    // Ensure annotation popup is a direct child of body (fixes position:fixed in stacking contexts)
    const _popup = document.getElementById('annotation-popup');
    if (_popup && _popup.parentElement !== document.body) {
        document.body.appendChild(_popup);
    }

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
            const refPreview = document.getElementById('ref-photo-preview');
            const refImg = document.getElementById('ref-photo-img');
            if (data.has_reference) {
                refStatusText.textContent = 'Reference Photo Active — Comparison enabled';
                refStatusText.style.color = 'var(--success)';
                clearRefBtn.classList.remove('hidden');
                if (data.ref_image_url && refPreview && refImg) {
                    refImg.src = data.ref_image_url + '?t=' + Date.now();
                    refPreview.classList.remove('hidden');
                    lucide.createIcons();
                }
            } else {
                refStatusText.textContent = 'No Reference — Heuristic detection only';
                refStatusText.style.color = 'var(--text-secondary)';
                clearRefBtn.classList.add('hidden');
                if (refPreview) refPreview.classList.add('hidden');
            }
        } catch (e) {
            console.error(e);
        }
    }
    checkReferenceStatus();

    clearRefBtn.addEventListener('click', async () => {
        await fetch('/api/clear_reference', { method: 'POST' });
        checkReferenceStatus();
        showToast('Reference Photo removed.', 'warning');
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
            showToast('Please upload an image file (JPG, PNG).', 'error');
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
        document.getElementById('loading-title').textContent = 'Analyzing shelf...';
        document.getElementById('loading-subtitle').textContent = 'Detecting products and checking shelf layout';
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
            showToast('An error occurred during analysis.', 'error');
        } finally {
            loadingOverlay.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            // If image already loaded before section became visible, re-setup click
            requestAnimationFrame(() => setupImageClick());
        }
    });

    setRefBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        document.getElementById('loading-title').textContent = 'Saving reference...';
        document.getElementById('loading-subtitle').textContent = 'Building shelf structure from this photo';
        loadingOverlay.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('file', currentFile);
            const response = await fetch('/api/set_reference', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('API request failed');
            const data = await response.json();

            if (data.status === 'success') {
                showToast('Reference Photo saved successfully!', 'success');
                checkReferenceStatus();
            } else {
                showToast('Failed to save reference: ' + data.message, 'error');
            }
        } catch (error) {
            console.error('Error:', error);
            showToast('An error occurred while setting reference.', 'error');
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    });

    // Close popup when clicking outside
    document.addEventListener('click', () => {
        document.getElementById('annotation-popup').style.display = 'none';
    });

    // Stores annotations to be drawn after section becomes visible
    let _complianceAnnotations = [];

    function displayComplianceImage(data) {
        const section = document.getElementById('compliance-section');

        // Show compliance image (original photo with colored boxes) if available
        if (data.compliance_image_url) {
            const img = document.getElementById('compliance-image');
            const container = document.getElementById('compliance-container');
            const annotations = data.compliance_annotations || [];

            img.onload = () => {
                container.querySelectorAll('canvas').forEach(c => c.remove());

                const canvas = document.createElement('canvas');
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair;border-radius:8px;';

                canvas.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const rect = canvas.getBoundingClientRect();
                    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
                    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

                    const ordered = [
                        ...annotations.filter(a => a.type === 'misplaced'),
                        ...annotations.filter(a => a.type === 'correct'),
                        ...annotations.filter(a => a.type === 'gap'),
                    ];
                    for (const ann of ordered) {
                        const b = ann.item.bbox;
                        if (b && x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2) {
                            showAnnotationPopup(ann.item, ann.type, e.clientX, e.clientY);
                            return;
                        }
                    }
                    document.getElementById('annotation-popup').style.display = 'none';
                });

                container.appendChild(canvas);
            };

            img.src = data.compliance_image_url + '?t=' + Date.now();
            section.style.display = 'block';
        }

        // Show planogram image with interactive canvas overlay
        if (!data.image_url) return;

        _complianceAnnotations = [];
        (data.gap_detections || []).forEach(item => {
            if (item.bbox) _complianceAnnotations.push({ item, type: 'gap' });
        });
        (data.correct_items || []).forEach(item => {
            if (item.bbox) _complianceAnnotations.push({ item, type: 'correct' });
        });
        (data.misplaced_items || []).forEach(item => {
            if (item.bbox) _complianceAnnotations.push({ item, type: 'misplaced' });
        });

        let planogramSection = document.getElementById('planogram-section');
        if (!planogramSection) {
            planogramSection = document.createElement('div');
            planogramSection.id = 'planogram-section';
            planogramSection.style.cssText = 'margin-top: 1rem;';
            planogramSection.innerHTML = `
                <h3 style="margin-bottom: 0.75rem; font-size: 1rem; color: var(--text-secondary);">Planogram Layout <span style="font-size:0.8rem; font-weight:400;">(click on a product to see details)</span></h3>
                <div id="planogram-container" style="position: relative; display: inline-block; max-width: 100%;">
                    <img id="planogram-image" src="" alt="Planogram Layout" style="display: block; max-height: 65vh; max-width: 100%; width: auto; border-radius: 8px;">
                </div>
            `;
            section.appendChild(planogramSection);
        }

        const planogramContainer = document.getElementById('planogram-container');
        const planogramImg = document.getElementById('planogram-image');

        planogramImg.onload = () => {
            planogramContainer.querySelectorAll('canvas').forEach(c => c.remove());

            const canvas = document.createElement('canvas');
            canvas.width = planogramImg.naturalWidth;
            canvas.height = planogramImg.naturalHeight;
            canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair;border-radius:8px;';

            const ctx = canvas.getContext('2d');
            ctx.lineWidth = 3;

            _complianceAnnotations.filter(a => a.type === 'correct').forEach(ann => {
                const b = ann.item.bbox;
                if (b) {
                    ctx.strokeStyle = '#00dc00';
                    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
                }
            });
            _complianceAnnotations.filter(a => a.type === 'misplaced').forEach(ann => {
                const b = ann.item.bbox;
                if (b) {
                    ctx.strokeStyle = '#dc0000';
                    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
                }
            });

            canvas.addEventListener('click', (e) => {
                e.stopPropagation();
                const rect = canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) * (canvas.width / rect.width);
                const y = (e.clientY - rect.top) * (canvas.height / rect.height);

                const ordered = [
                    ..._complianceAnnotations.filter(a => a.type === 'misplaced'),
                    ..._complianceAnnotations.filter(a => a.type === 'correct'),
                    ..._complianceAnnotations.filter(a => a.type === 'gap'),
                ];
                for (const ann of ordered) {
                    const b = ann.item.bbox;
                    if (b && x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2) {
                        showAnnotationPopup(ann.item, ann.type, e.clientX, e.clientY);
                        return;
                    }
                }
                document.getElementById('annotation-popup').style.display = 'none';
            });

            planogramContainer.appendChild(canvas);
        };

        planogramImg.src = data.image_url + '?t=' + Date.now();
        planogramSection.style.display = 'block';
    }

    function setupImageClick() { /* no-op */ }

    function showAnnotationPopup(item, type, x, y) {
        const popup = document.getElementById('annotation-popup');
        const content = document.getElementById('popup-content');

        let html = '';
        if (type === 'correct') {
            html = `
                <div style="color:var(--success);font-weight:600;margin-bottom:6px;">✓ Correct Item</div>
                <div style="font-size:0.85rem;color:var(--text-secondary);">${item.label || item.detected_label || 'Product'}</div>
                <div style="font-size:0.8rem;color:#888;margin-top:4px;">Shelf ${item.shelf ?? item.detected_shelf ?? '-'}</div>
            `;
        } else if (type === 'misplaced') {
            const found = item.detected_label || item.label || 'Unknown Item';
            const hasExpected = item.expected_label && item.expected_label !== 'Unknown';
            if (hasExpected) {
                html = `
                    <div style="color:var(--danger);font-weight:600;margin-bottom:8px;">⚠ Misplaced Product</div>
                    <div style="font-size:0.85rem;margin-bottom:4px;"><span style="color:var(--success);font-weight:500;">Expected:</span> <span style="color:var(--text-secondary);">${item.expected_label}</span></div>
                    <div style="font-size:0.85rem;margin-bottom:4px;"><span style="color:var(--danger);font-weight:500;">Found:</span> <span style="color:var(--text-primary);">${found}</span></div>
                    ${item.detail_msg ? `<div style="font-size:0.75rem;color:#888;font-style:italic;margin-top:4px;">${item.detail_msg}</div>` : ''}
                `;
            } else {
                html = `
                    <div style="color:var(--danger);font-weight:600;margin-bottom:8px;">⚠ Unexpected Product</div>
                    <div style="font-size:0.85rem;margin-bottom:4px;"><span style="color:var(--danger);font-weight:500;">Found:</span> <span style="color:var(--text-primary);">${found}</span></div>
                    <div style="font-size:0.75rem;color:#888;font-style:italic;margin-top:4px;">${item.detail_msg || 'This product is not in the reference image'}</div>
                    <div style="font-size:0.8rem;color:#888;margin-top:4px;">Shelf ${item.detected_shelf ?? '-'}</div>
                `;
            }
        } else if (type === 'gap') {
            html = `
                <div style="color:var(--warning);font-weight:600;margin-bottom:8px;">◻ Physical Gap Detected</div>
                <div style="font-size:0.85rem;color:var(--text-secondary);">Empty space on shelf</div>
                <div style="font-size:0.8rem;color:#888;margin-top:4px;">Shelf ${item.expected_shelf ?? '-'}</div>
                ${item.label && item.label !== 'Empty Space' ? `<div style="font-size:0.8rem;margin-top:4px;"><span style="color:var(--warning);">Space for:</span> ${item.label}</div>` : ''}
            `;
        }

        content.innerHTML = html;

        const popupW = 300, popupH = 130;
        let left = x + 14;
        let top = y - 20;
        if (left + popupW > window.innerWidth) left = x - popupW - 14;
        if (top + popupH > window.innerHeight) top = window.innerHeight - popupH - 12;

        popup.style.left = left + 'px';
        popup.style.top = top + 'px';
        popup.style.display = 'block';

        // Hide popup when user scrolls
        const hidePopupOnScroll = () => {
            popup.style.display = 'none';
            window.removeEventListener('scroll', hidePopupOnScroll, true);
        };
        window.removeEventListener('scroll', hidePopupOnScroll, true); // clear any previous
        window.addEventListener('scroll', hidePopupOnScroll, true);
    }

    function displayResults(data) {
        scoreBadge.textContent = `Analysis Complete`;
        scoreBadge.style.color = 'var(--text-primary)';
        scoreBadge.style.borderColor = 'var(--border-color)';
        scoreBadge.style.background = 'var(--surface-color)';

        totalItems.textContent = data.total_items;

        const gapCount = data.gap_detections ? data.gap_detections.length : 0;
        const missingItemCount = data.missing_items ? data.missing_items.length : 0;
        const misplacedItemCount = data.misplaced_items ? data.misplaced_items.length : 0;

        missingItems.textContent = gapCount;
        misplacedItems.textContent = misplacedItemCount;

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
                        <strong style="display:block; margin-bottom:4px;">Product not found on shelf</strong>
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
                        <strong style="display:block; margin-bottom:6px;">Wrong product in this slot</strong>
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
                const spaceLabel = item.label && item.label !== 'Empty Space' ? item.label : null;
                li.innerHTML = `
                    <div>
                        <strong style="display:block; margin-bottom:4px;">Empty space — product may be out of stock</strong>
                        ${spaceLabel ? `<div style="font-size: 0.875rem; color: var(--text-secondary);"><span style="color: var(--warning); font-weight: 500;">Expected here:</span> ${spaceLabel}</div>` : `<div style="font-size: 0.875rem; color: var(--text-secondary);">No product detected in this slot</div>`}
                    </div>
                    <div style="text-align: right; color: var(--text-secondary); font-size: 0.875rem;">
                        Shelf ${item.expected_shelf}
                    </div>
                `;
                issueList.appendChild(li);
            });
        }

        lucide.createIcons();
        displayComplianceImage(data); // sets img.src and stores annotations; overlays built after section visible
    }
});
