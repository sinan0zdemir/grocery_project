// Global filter function for metric cards (dashboard)
window.filterIssues = function (filterType) {
    const items = document.querySelectorAll('#issue-list .issue-item');
    items.forEach(item => {
        if (filterType === 'all') {
            item.style.display = 'flex';
        } else {
            item.style.display = item.classList.contains(filterType) ? 'flex' : 'none';
        }
    });
};

// Global filter function for comparison results
window.filterCompareIssues = function (filterType) {
    const items = document.querySelectorAll('#compare-issue-list .issue-item');
    items.forEach(item => {
        if (filterType === 'all') {
            item.style.display = 'flex';
        } else {
            item.style.display = item.classList.contains(filterType) ? 'flex' : 'none';
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
    const uploadContent = document.getElementById('upload-content');
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
        dropZone.classList.remove('has-preview');
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
            dropZone.classList.add('has-preview');
            analyzeBtn.disabled = false;
            setRefBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // --- Dashboard API Integration ---
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
                updateRefCountBadge();
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
        renderIssueList(issueList, data);
    }

    // =====================================================
    // REFERENCES TAB
    // =====================================================
    const refListBody = document.getElementById('ref-list-body');
    const refEmptyState = document.getElementById('ref-empty-state');
    const refCountBadge = document.getElementById('ref-count-badge');

    // Comparison elements
    const compareRefImg = document.getElementById('compare-ref-img');
    const compareRefPlaceholder = document.getElementById('compare-ref-placeholder');
    const compareRefWrap = document.getElementById('compare-ref-wrap');
    const compareNewImg = document.getElementById('compare-new-img');
    const compareUploadPlaceholder = document.getElementById('compare-upload-placeholder');
    const compareDropZone = document.getElementById('compare-drop-zone');
    const compareFileInput = document.getElementById('compare-file-input');
    const compareBrowseBtn = document.getElementById('compare-browse-btn');
    const compareRemoveBtn = document.getElementById('compare-remove-btn');
    const compareRemoveOverlay = document.getElementById('compare-remove-overlay');
    const compareAnalyzeBtn = document.getElementById('compare-analyze-btn');
    const compareStatusBadge = document.getElementById('compare-status-badge');
    const compareResultsSection = document.getElementById('compare-results-section');

    let selectedRefId = null;
    let compareFile = null;

    // --- Load References ---
    async function loadReferencesList() {
        try {
            const res = await fetch('/api/list_references');
            const data = await res.json();
            const refs = data.references || [];

            // Clear existing cards (keep empty state)
            const existingCards = refListBody.querySelectorAll('.ref-card');
            existingCards.forEach(c => c.remove());

            if (refs.length === 0) {
                refEmptyState.classList.remove('hidden');
            } else {
                refEmptyState.classList.add('hidden');
                refs.forEach(ref => {
                    const card = createRefCard(ref);
                    refListBody.appendChild(card);
                });
            }

            // Update badge
            refCountBadge.textContent = refs.length;
            if (refs.length > 0) {
                refCountBadge.classList.remove('hidden');
            } else {
                refCountBadge.classList.add('hidden');
            }

            lucide.createIcons();
        } catch (e) {
            console.error('Failed to load references:', e);
        }
    }

    function createRefCard(ref) {
        const card = document.createElement('div');
        card.className = 'ref-card' + (ref.id === selectedRefId ? ' selected' : '');
        card.dataset.refId = ref.id;

        const createdDate = ref.created_at
            ? new Date(ref.created_at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })
            : 'Unknown date';

        card.innerHTML = `
            <div class="ref-card-thumb">
                <img src="${ref.image_url}" alt="${ref.name}" loading="lazy">
            </div>
            <div class="ref-card-info">
                <div class="ref-card-name">
                    ${ref.name}
                    ${ref.active ? '<span class="ref-active-badge">Active</span>' : ''}
                </div>
                <div class="ref-card-date">${createdDate}</div>
            </div>
            <div class="ref-card-actions">
                ${!ref.active ? `<button class="btn btn-primary btn-sm ref-activate-btn" data-ref-id="${ref.id}" title="Activate this reference">
                    <i data-lucide="check-circle"></i> Activate
                </button>` : ''}
                <button class="btn btn-danger-ghost btn-sm ref-delete-btn" data-ref-id="${ref.id}" title="Delete this reference">
                    <i data-lucide="trash-2"></i>
                </button>
            </div>
        `;

        // Select card on click (but not on button click)
        card.addEventListener('click', (e) => {
            if (e.target.closest('.ref-activate-btn') || e.target.closest('.ref-delete-btn')) return;
            selectReference(ref);
        });

        // Activate button
        const activateBtn = card.querySelector('.ref-activate-btn');
        if (activateBtn) {
            activateBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                await activateReference(ref.id);
            });
        }

        // Delete button
        const deleteBtn = card.querySelector('.ref-delete-btn');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                if (confirm(`Delete reference "${ref.name}"?`)) {
                    await deleteReference(ref.id);
                }
            });
        }

        return card;
    }

    function selectReference(ref) {
        selectedRefId = ref.id;

        // Update card selection state
        const cards = refListBody.querySelectorAll('.ref-card');
        cards.forEach(c => c.classList.toggle('selected', c.dataset.refId === ref.id));

        // Show reference image
        compareRefImg.src = ref.image_url;
        compareRefImg.classList.remove('hidden');
        compareRefPlaceholder.classList.add('hidden');
        compareRefWrap.classList.add('has-image');

        // Update status
        compareStatusBadge.textContent = `Reference: ${ref.name}`;
        compareStatusBadge.style.color = 'var(--primary)';
        compareStatusBadge.style.borderColor = 'rgba(79, 70, 229, 0.3)';
        compareStatusBadge.style.background = 'rgba(79, 70, 229, 0.15)';

        updateCompareAnalyzeState();
    }

    async function activateReference(refId) {
        try {
            const formData = new FormData();
            formData.append('ref_id', refId);
            const res = await fetch('/api/activate_reference', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.status === 'success') {
                loadReferencesList();
                checkReferenceStatus();
            } else {
                alert('Failed to activate: ' + data.message);
            }
        } catch (e) {
            console.error(e);
        }
    }

    async function deleteReference(refId) {
        try {
            const formData = new FormData();
            formData.append('ref_id', refId);
            const res = await fetch('/api/clear_reference', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.status === 'success') {
                if (selectedRefId === refId) {
                    selectedRefId = null;
                    resetCompareView();
                }
                loadReferencesList();
                checkReferenceStatus();
            } else {
                alert('Failed to delete: ' + data.message);
            }
        } catch (e) {
            console.error(e);
        }
    }

    function resetCompareView() {
        compareRefImg.classList.add('hidden');
        compareRefPlaceholder.classList.remove('hidden');
        compareRefWrap.classList.remove('has-image');
        compareStatusBadge.textContent = 'Select a reference shelf first';
        compareStatusBadge.style.color = '';
        compareStatusBadge.style.borderColor = '';
        compareStatusBadge.style.background = '';
        updateCompareAnalyzeState();
    }

    async function updateRefCountBadge() {
        try {
            const res = await fetch('/api/list_references');
            const data = await res.json();
            const count = (data.references || []).length;
            refCountBadge.textContent = count;
            if (count > 0) {
                refCountBadge.classList.remove('hidden');
            } else {
                refCountBadge.classList.add('hidden');
            }
        } catch (e) {
            console.error(e);
        }
    }
    updateRefCountBadge();

    // --- Compare: Upload new image ---
    compareBrowseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        compareFileInput.click();
    });

    compareDropZone.addEventListener('click', (e) => {
        if (!compareFile && !e.target.closest('.btn')) {
            compareFileInput.click();
        }
    });

    compareFileInput.addEventListener('change', function () {
        if (this.files.length > 0) handleCompareFile(this.files[0]);
    });

    // Drag & Drop on compare zone
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        compareDropZone.addEventListener(eventName, preventDefaults, false);
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        compareDropZone.addEventListener(eventName, () => compareDropZone.classList.add('dragover'), false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        compareDropZone.addEventListener(eventName, () => compareDropZone.classList.remove('dragover'), false);
    });
    compareDropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleCompareFile(files[0]);
    }, false);

    function handleCompareFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (JPG, PNG).');
            return;
        }
        compareFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            compareNewImg.src = e.target.result;
            compareNewImg.classList.remove('hidden');
            compareUploadPlaceholder.classList.add('hidden');
            compareRemoveOverlay.classList.remove('hidden');
            compareDropZone.classList.add('has-image');
            updateCompareAnalyzeState();
        };
        reader.readAsDataURL(file);
    }

    compareRemoveBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        compareFile = null;
        compareNewImg.src = '';
        compareNewImg.classList.add('hidden');
        compareUploadPlaceholder.classList.remove('hidden');
        compareRemoveOverlay.classList.add('hidden');
        compareDropZone.classList.remove('has-image');
        compareFileInput.value = '';
        updateCompareAnalyzeState();
    });

    function updateCompareAnalyzeState() {
        compareAnalyzeBtn.disabled = !(selectedRefId && compareFile);
    }

    // --- Compare: Analyze ---
    compareAnalyzeBtn.addEventListener('click', async () => {
        if (!selectedRefId || !compareFile) return;

        // First, activate the selected reference
        const activateForm = new FormData();
        activateForm.append('ref_id', selectedRefId);
        try {
            loadingOverlay.classList.remove('hidden');
            compareResultsSection.classList.add('hidden');

            await fetch('/api/activate_reference', { method: 'POST', body: activateForm });

            // Then analyze
            const formData = new FormData();
            formData.append('file', compareFile);
            const response = await fetch('/api/analyze', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('API request failed');
            const data = await response.json();
            displayCompareResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during comparison analysis.');
        } finally {
            loadingOverlay.classList.add('hidden');
            compareResultsSection.classList.remove('hidden');
            compareResultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });

    function displayCompareResults(data) {
        const cScoreBadge = document.getElementById('compare-score-badge');
        const cTotalItems = document.getElementById('compare-total-items');
        const cMissingItems = document.getElementById('compare-missing-items');
        const cMisplacedItems = document.getElementById('compare-misplaced-items');
        const cIssueList = document.getElementById('compare-issue-list');

        cScoreBadge.textContent = 'Comparison Complete';
        cScoreBadge.style.color = 'var(--text-primary)';
        cScoreBadge.style.borderColor = 'var(--border-color)';
        cScoreBadge.style.background = 'var(--surface-color)';

        cTotalItems.textContent = data.total_items + " Items";

        const gapCount = data.gap_detections ? data.gap_detections.length : 0;
        const misplacedItemCount = data.misplaced_items ? data.misplaced_items.length : 0;

        cMissingItems.textContent = gapCount + " Items";
        cMisplacedItems.textContent = misplacedItemCount + " Items";

        renderIssueList(cIssueList, data);
    }

    // =====================================================
    // SHARED – Render Issue List
    // =====================================================
    function renderIssueList(container, data) {
        container.innerHTML = '';
        const gapCount = data.gap_detections ? data.gap_detections.length : 0;
        const missingItemCount = data.missing_items ? data.missing_items.length : 0;
        const misplacedItemCount = data.misplaced_items ? data.misplaced_items.length : 0;

        if (gapCount === 0 && missingItemCount === 0 && misplacedItemCount === 0) {
            container.innerHTML = `<li class="empty-state" style="border: 1px solid var(--success); color: var(--success);">
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
                container.appendChild(li);
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
                container.appendChild(li);
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
                container.appendChild(li);
            });
        }

        lucide.createIcons();
        displayComplianceImage(data); // sets img.src and stores annotations; overlays built after section visible
    }
});
