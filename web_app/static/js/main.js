/* ============================================================
   PlanoAI — Frontend
   ============================================================ */

window.filterIssues = function (filterType) {
    const items = document.querySelectorAll('#issue-list .issue-item');
    items.forEach(item => {
        if (filterType === 'all' || item.classList.contains(filterType)) {
            item.style.display = 'flex';
        } else {
            item.style.display = 'none';
        }
    });
};

/* ---------- Toast ---------- */
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = {
        success: 'check-circle',
        error: 'x-circle',
        warning: 'alert-triangle',
        info: 'info'
    };
    toast.innerHTML = `
        <i data-lucide="${icons[type] || icons.info}" class="toast-icon"></i>
        <span>${message}</span>
    `;
    container.appendChild(toast);
    if (window.lucide) lucide.createIcons();
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3600);
}

/* ---------- Confirm modal ---------- */
function showConfirm({ title, message, okLabel = 'Confirm', okClass = 'btn-danger' }) {
    return new Promise((resolve) => {
        const modal = document.getElementById('confirm-modal');
        document.getElementById('confirm-title').textContent = title;
        document.getElementById('confirm-message').textContent = message;
        const okBtn = document.getElementById('confirm-ok');
        okBtn.textContent = okLabel;
        okBtn.className = `btn ${okClass}`;

        const cancelBtn = document.getElementById('confirm-cancel');

        const cleanup = (result) => {
            modal.classList.add('hidden');
            okBtn.removeEventListener('click', okHandler);
            cancelBtn.removeEventListener('click', cancelHandler);
            modal.removeEventListener('click', backdropHandler);
            resolve(result);
        };
        const okHandler = () => cleanup(true);
        const cancelHandler = () => cleanup(false);
        const backdropHandler = (e) => { if (e.target === modal) cleanup(false); };

        okBtn.addEventListener('click', okHandler);
        cancelBtn.addEventListener('click', cancelHandler);
        modal.addEventListener('click', backdropHandler);
        modal.classList.remove('hidden');
    });
}

/* ---------- Routing ---------- */
const VALID_ROUTES = ['dashboard', 'references'];

function navigateTo(route, { skipHash = false } = {}) {
    if (!VALID_ROUTES.includes(route)) route = 'dashboard';
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const page = document.getElementById(`page-${route}`);
    if (page) page.classList.add('active');
    const navItem = document.querySelector(`.nav-item[data-route="${route}"]`);
    if (navItem) navItem.classList.add('active');
    if (route === 'references' && typeof loadReferences === 'function') {
        loadReferences();
    }
    if (!skipHash) {
        const hash = route === 'dashboard' ? '' : `#${route}`;
        if (window.location.hash !== hash) {
            history.replaceState(null, '', window.location.pathname + hash);
        }
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function routeFromHash() {
    const hash = (window.location.hash || '').replace(/^#/, '');
    return VALID_ROUTES.includes(hash) ? hash : 'dashboard';
}

window.addEventListener('hashchange', () => {
    navigateTo(routeFromHash(), { skipHash: true });
});

/* ---------- Date formatting ---------- */
function formatDate(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    return d.toLocaleDateString(undefined, { day: '2-digit', month: 'short', year: 'numeric' }) +
           ' · ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

/* ---------- References management ---------- */
let _referencesCache = [];

async function fetchReferences() {
    try {
        const res = await fetch('/api/references');
        const data = await res.json();
        _referencesCache = data.references || [];
        return _referencesCache;
    } catch (e) {
        console.error(e);
        return [];
    }
}

function renderReferences(refs) {
    const grid = document.getElementById('references-grid');
    const empty = document.getElementById('references-empty');
    const summary = document.getElementById('ref-summary');
    const deactivateBtn = document.getElementById('deactivate-all-btn');
    const navBadge = document.getElementById('nav-ref-count');

    if (refs.length === 0) {
        grid.innerHTML = '';
        empty.classList.remove('hidden');
        summary.textContent = 'No references saved yet.';
        deactivateBtn.style.display = 'none';
        navBadge.style.display = 'none';
        return;
    }

    empty.classList.add('hidden');
    const activeRef = refs.find(r => r.active);
    summary.innerHTML = activeRef
        ? `<span class="dot dot-active"></span> <strong style="color:var(--text-primary);">${escapeHtml(activeRef.name)}</strong> is active`
        : `<span class="dot dot-idle"></span> ${refs.length} reference${refs.length === 1 ? '' : 's'} stored · none active`;
    deactivateBtn.style.display = activeRef ? 'inline-flex' : 'none';

    navBadge.style.display = 'inline-block';
    navBadge.textContent = refs.length;

    grid.innerHTML = refs.map(ref => `
        <article class="ref-card ${ref.active ? 'is-active' : ''}" data-id="${ref.id}">
            <div class="ref-card-image">
                ${ref.active ? `<span class="ref-card-active-pill"><i data-lucide="check"></i> Active</span>` : ''}
                <img src="${ref.image_url}" alt="${escapeHtml(ref.name)}" loading="lazy">
            </div>
            <div class="ref-card-body">
                <div class="ref-card-name" title="${escapeHtml(ref.name)}">${escapeHtml(ref.name)}</div>
                <div class="ref-card-meta">
                    <i data-lucide="calendar"></i>
                    <span>${formatDate(ref.created_at)}</span>
                </div>
                <div class="ref-card-actions">
                    ${ref.active
                        ? `<button class="btn btn-ghost" data-action="deactivate"><i data-lucide="power-off"></i> Deactivate</button>`
                        : `<button class="btn btn-success" data-action="activate"><i data-lucide="zap"></i> Activate</button>`
                    }
                    <button class="btn btn-ghost" data-action="delete" title="Delete reference" style="flex:0 0 auto;">
                        <i data-lucide="trash-2"></i>
                    </button>
                </div>
            </div>
        </article>
    `).join('');

    if (window.lucide) lucide.createIcons();

    grid.querySelectorAll('.ref-card').forEach(card => {
        const id = card.dataset.id;
        card.querySelector('[data-action="activate"]')?.addEventListener('click', () => activateReference(id));
        card.querySelector('[data-action="deactivate"]')?.addEventListener('click', () => deactivateReferences());
        card.querySelector('[data-action="delete"]')?.addEventListener('click', () => deleteReference(id));
    });
}

function escapeHtml(s) {
    if (s == null) return '';
    return String(s).replace(/[&<>"']/g, c =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
    );
}

// Strip phrasings the user wants out: shelf-specific assertions and "Golden Image"
function sanitizeMsg(s) {
    if (!s) return s;
    return String(s)
        .replace(/Belongs on Shelf [^,.]*/gi, 'Different product expected')
        .replace(/Golden Image/gi, 'reference image')
        .replace(/—/g, ',');
}

async function loadReferences() {
    const refs = await fetchReferences();
    renderReferences(refs);
    updateDashboardRefIndicator();
}

async function activateReference(id) {
    try {
        const res = await fetch(`/api/references/${id}/activate`, { method: 'POST' });
        if (!res.ok) throw new Error('Failed');
        const data = await res.json();
        showToast(`Activated "${data.reference?.name || 'reference'}"`, 'success');
        await loadReferences();
    } catch (e) {
        console.error(e);
        showToast('Could not activate reference.', 'error');
    }
}

async function deactivateReferences() {
    const ok = await showConfirm({
        title: 'Deactivate reference?',
        message: 'PlanoAI will fall back to heuristic-only detection until a reference is activated again.',
        okLabel: 'Deactivate',
        okClass: 'btn-ghost'
    });
    if (!ok) return;
    try {
        await fetch('/api/references/deactivate', { method: 'POST' });
        showToast('Reference deactivated.', 'warning');
        await loadReferences();
    } catch (e) {
        showToast('Could not deactivate.', 'error');
    }
}

async function deleteReference(id) {
    const ref = _referencesCache.find(r => r.id === id);
    const ok = await showConfirm({
        title: 'Delete reference?',
        message: `"${ref?.name || 'This reference'}" will be permanently removed. This cannot be undone.`,
        okLabel: 'Delete',
        okClass: 'btn-danger'
    });
    if (!ok) return;
    try {
        const res = await fetch(`/api/references/${id}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Failed');
        showToast('Reference deleted.', 'success');
        await loadReferences();
    } catch (e) {
        showToast('Could not delete reference.', 'error');
    }
}

/* ---------- Dashboard reference indicator ---------- */
function updateDashboardRefIndicator() {
    const pill = document.getElementById('dash-ref-pill');
    const label = pill.querySelector('.status-label');
    const dot = pill.querySelector('.dot');
    const thumb = document.getElementById('ref-state-thumb');
    const stateLabel = document.getElementById('ref-state-label');
    const stateSub = document.getElementById('ref-state-sub');
    const footerStatus = document.getElementById('footer-status');

    // Active reference dashboard card
    const activeCard = document.getElementById('active-ref-card');
    const activeImg = document.getElementById('active-ref-img');
    const activeName = document.getElementById('active-ref-name');
    const activeDate = document.getElementById('active-ref-date');

    // Upload section headline switches depending on whether a reference is set
    const uploadHeadline = document.getElementById('upload-headline');
    const sectionLabel = document.getElementById('upload-section-label');

    const active = _referencesCache.find(r => r.active);

    if (active) {
        pill.classList.add('is-active');
        dot.className = 'dot dot-active';
        label.textContent = 'Reference active';

        thumb.innerHTML = `<img src="${active.image_url}" alt="">`;
        stateLabel.textContent = active.name;
        stateSub.innerHTML = `Active reference · <a href="#" data-route="references" class="link">Manage</a>`;

        footerStatus.innerHTML = `<span class="dot dot-active"></span><span>Comparison enabled</span>`;

        if (activeCard) {
            activeCard.classList.remove('hidden');
            activeImg.src = active.image_url;
            activeName.textContent = active.name;
            activeDate.textContent = `Saved on ${formatDate(active.created_at)}`;
        }
        if (uploadHeadline) uploadHeadline.textContent = 'Drop the shelf photo to compare';
        if (sectionLabel) sectionLabel.querySelector('h3').textContent = 'Upload the shelf photo to compare';
        if (sectionLabel) sectionLabel.querySelector('.eyebrow').textContent = 'Step 2';
    } else {
        pill.classList.remove('is-active');
        dot.className = 'dot dot-idle';
        label.textContent = 'No reference';

        thumb.innerHTML = `<i data-lucide="image-off"></i>`;
        stateLabel.textContent = 'No active reference';
        stateSub.innerHTML = `Heuristic detection only · <a href="#" data-route="references" class="link">Manage references</a>`;

        footerStatus.innerHTML = `<span class="dot dot-idle"></span><span>Heuristic mode</span>`;

        if (activeCard) activeCard.classList.add('hidden');
        if (uploadHeadline) uploadHeadline.textContent = 'Drop your shelf photo here';
        if (sectionLabel) sectionLabel.querySelector('h3').textContent = 'Upload a shelf photo to analyze';
        if (sectionLabel) sectionLabel.querySelector('.eyebrow').textContent = 'Get started';
    }

    if (window.lucide) lucide.createIcons();

    // re-bind data-route links (they were re-rendered)
    document.querySelectorAll('[data-route]').forEach(el => {
        el.onclick = (e) => {
            e.preventDefault();
            navigateTo(el.dataset.route);
        };
    });
}

/* ============================================================
   Main DOMContentLoaded
   ============================================================ */
document.addEventListener('DOMContentLoaded', async () => {
    // Ensure popup is direct child of body
    const popup = document.getElementById('annotation-popup');
    if (popup && popup.parentElement !== document.body) {
        document.body.appendChild(popup);
    }

    // Move modals to body too, so they're not constrained by stacking contexts
    ['new-ref-modal', 'confirm-modal'].forEach(id => {
        const m = document.getElementById(id);
        if (m && m.parentElement !== document.body) document.body.appendChild(m);
    });

    /* ---------- Wire routing ---------- */
    document.querySelectorAll('[data-route]').forEach(el => {
        el.addEventListener('click', (e) => {
            e.preventDefault();
            navigateTo(el.dataset.route);
        });
    });

    /* ---------- Popup close ---------- */
    document.getElementById('popup-close').addEventListener('click', (e) => {
        e.stopPropagation();
        popup.style.display = 'none';
    });

    /* ---------- Mouse-follow gradient on upload area ---------- */
    const dropZone = document.getElementById('drop-zone');
    dropZone.addEventListener('mousemove', (e) => {
        const rect = dropZone.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        dropZone.style.setProperty('--mx', `${x}%`);
        dropZone.style.setProperty('--my', `${y}%`);
    });

    /* ---------- Dashboard upload elements ---------- */
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewContainer = document.getElementById('preview-container');
    const uploadContent = document.querySelector('.upload-content');
    const imagePreview = document.getElementById('image-preview');
    const removeImgBtn = document.getElementById('remove-img-btn');
    const replaceImgBtn = document.getElementById('replace-img-btn');
    const analyzeBtn = document.getElementById('analyze-btn');

    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsSection = document.getElementById('results-section');

    const scoreBadge = document.getElementById('score-badge');
    const totalItems = document.getElementById('total-items');
    const missingItems = document.getElementById('missing-items');
    const misplacedItems = document.getElementById('misplaced-items');
    const issueList = document.getElementById('issue-list');

    let currentFile = null;

    /* ---------- Drag & drop dashboard ---------- */
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev =>
        dropZone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); })
    );
    ['dragenter', 'dragover'].forEach(ev =>
        dropZone.addEventListener(ev, () => dropZone.classList.add('dragover'))
    );
    ['dragleave', 'drop'].forEach(ev =>
        dropZone.addEventListener(ev, () => dropZone.classList.remove('dragover'))
    );
    dropZone.addEventListener('drop', e => {
        const f = e.dataTransfer.files[0];
        if (f) handleDashboardFile(f);
    });

    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function () {
        if (this.files.length > 0) handleDashboardFile(this.files[0]);
    });

    removeImgBtn.addEventListener('click', e => {
        e.stopPropagation();
        clearDashboardImage();
    });

    replaceImgBtn.addEventListener('click', e => {
        e.stopPropagation();
        fileInput.value = '';
        fileInput.click();
    });

    function clearDashboardImage() {
        currentFile = null;
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        analyzeBtn.disabled = true;
        fileInput.value = '';
    }

    function handleDashboardFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please upload an image file (JPG or PNG).', 'error');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            imagePreview.src = e.target.result;
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    /* ---------- Analyze ---------- */
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        document.getElementById('loading-title').textContent = 'Analyzing shelf…';
        document.getElementById('loading-subtitle').textContent = 'Detecting products and checking layout';
        loadingOverlay.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('file', currentFile);
            const res = await fetch('/api/analyze', { method: 'POST', body: formData });
            if (!res.ok) throw new Error('API request failed');
            const data = await res.json();
            displayResults(data);
            resultsSection.classList.remove('hidden');
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch (e) {
            console.error(e);
            showToast('An error occurred during analysis.', 'error');
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    });

    /* ---------- References page: Add modal ---------- */
    const newRefModal = document.getElementById('new-ref-modal');
    const modalDropZone = document.getElementById('modal-drop-zone');
    const modalFileInput = document.getElementById('modal-file-input');
    const modalUploadEmpty = document.getElementById('modal-upload-empty');
    const modalUploadPreview = document.getElementById('modal-upload-preview');
    const modalPreviewImg = document.getElementById('modal-preview-img');
    const modalChangeBtn = document.getElementById('modal-change-btn');
    const modalSaveBtn = document.getElementById('modal-save-btn');
    const modalCancelBtn = document.getElementById('modal-cancel-btn');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const refNameInput = document.getElementById('ref-name-input');

    let modalFile = null;

    function openNewRefModal() {
        modalFile = null;
        refNameInput.value = '';
        modalUploadPreview.classList.add('hidden');
        modalUploadEmpty.classList.remove('hidden');
        modalSaveBtn.disabled = true;
        newRefModal.classList.remove('hidden');
        if (window.lucide) lucide.createIcons();
    }

    function closeNewRefModal() { newRefModal.classList.add('hidden'); }

    document.getElementById('add-ref-btn').addEventListener('click', openNewRefModal);
    document.getElementById('empty-add-ref-btn').addEventListener('click', openNewRefModal);
    modalCancelBtn.addEventListener('click', closeNewRefModal);
    modalCloseBtn.addEventListener('click', closeNewRefModal);
    newRefModal.addEventListener('click', e => { if (e.target === newRefModal) closeNewRefModal(); });

    modalDropZone.addEventListener('click', e => {
        if (e.target.closest('.modal-preview-change')) return;
        if (!modalFile) modalFileInput.click();
    });
    modalChangeBtn.addEventListener('click', e => {
        e.stopPropagation();
        modalFileInput.click();
    });
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev =>
        modalDropZone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); })
    );
    ['dragenter', 'dragover'].forEach(ev =>
        modalDropZone.addEventListener(ev, () => modalDropZone.classList.add('dragover'))
    );
    ['dragleave', 'drop'].forEach(ev =>
        modalDropZone.addEventListener(ev, () => modalDropZone.classList.remove('dragover'))
    );
    modalDropZone.addEventListener('drop', e => {
        const f = e.dataTransfer.files[0];
        if (f) handleModalFile(f);
    });
    modalFileInput.addEventListener('change', function () {
        if (this.files.length > 0) handleModalFile(this.files[0]);
    });

    function handleModalFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please upload an image file.', 'error');
            return;
        }
        modalFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            modalPreviewImg.src = e.target.result;
            modalUploadEmpty.classList.add('hidden');
            modalUploadPreview.classList.remove('hidden');
            modalSaveBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    modalSaveBtn.addEventListener('click', async () => {
        if (!modalFile) return;
        modalSaveBtn.disabled = true;
        document.getElementById('loading-title').textContent = 'Saving reference…';
        document.getElementById('loading-subtitle').textContent = 'Building shelf schema from this photo';
        loadingOverlay.classList.remove('hidden');
        closeNewRefModal();

        try {
            const form = new FormData();
            form.append('file', modalFile);
            const name = refNameInput.value.trim();
            const url = name ? `/api/references?name=${encodeURIComponent(name)}` : '/api/references';
            const res = await fetch(url, { method: 'POST', body: form });
            const data = await res.json();
            if (data.status === 'success') {
                showToast('Reference saved successfully.', 'success');
                await loadReferences();
            } else {
                showToast(data.message || 'Could not save reference.', 'error');
            }
        } catch (e) {
            console.error(e);
            showToast('Could not save reference.', 'error');
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    });

    /* ---------- Footer deactivate button on References page ---------- */
    document.getElementById('deactivate-all-btn').addEventListener('click', () => deactivateReferences());

    /* ---------- Dashboard deactivate ---------- */
    document.getElementById('dash-deactivate-btn').addEventListener('click', () => deactivateReferences());

    /* ---------- Click outside popup ---------- */
    document.addEventListener('click', () => {
        document.getElementById('annotation-popup').style.display = 'none';
    });

    /* ---------- Initial load ---------- */
    await loadReferences();
    // Honor URL hash on initial load
    navigateTo(routeFromHash(), { skipHash: true });

    /* ============================================================
       Results rendering (kept from original, refined)
       ============================================================ */
    let _complianceAnnotations = [];

    function displayComplianceImage(data) {
        const section = document.getElementById('compliance-section');

        if (data.compliance_image_url) {
            const img = document.getElementById('compliance-image');
            const container = document.getElementById('compliance-container');
            const annotations = data.compliance_annotations || [];

            img.onload = () => {
                container.querySelectorAll('canvas').forEach(c => c.remove());
                const canvas = document.createElement('canvas');
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair;border-radius:inherit;';

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

        if (!data.image_url) return;

        _complianceAnnotations = [];
        (data.gap_detections || []).forEach(item => { if (item.bbox) _complianceAnnotations.push({ item, type: 'gap' }); });
        (data.correct_items || []).forEach(item => { if (item.bbox) _complianceAnnotations.push({ item, type: 'correct' }); });
        (data.misplaced_items || []).forEach(item => { if (item.bbox) _complianceAnnotations.push({ item, type: 'misplaced' }); });

        let planogramSection = document.getElementById('planogram-section');
        if (!planogramSection) {
            planogramSection = document.createElement('div');
            planogramSection.id = 'planogram-section';
            planogramSection.style.marginTop = '1.5rem';
            planogramSection.innerHTML = `
                <div class="section-label">
                    <h3>Planogram Layout</h3>
                    <span class="hint">Click any product for details</span>
                </div>
                <div id="planogram-container" class="image-container">
                    <img id="planogram-image" src="" alt="Planogram Layout">
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
            canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair;border-radius:inherit;';
            const ctx = canvas.getContext('2d');
            ctx.lineWidth = 3;

            _complianceAnnotations.filter(a => a.type === 'correct').forEach(ann => {
                const b = ann.item.bbox;
                if (b) { ctx.strokeStyle = '#34D399'; ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1); }
            });
            _complianceAnnotations.filter(a => a.type === 'misplaced').forEach(ann => {
                const b = ann.item.bbox;
                if (b) { ctx.strokeStyle = '#F87171'; ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1); }
            });

            canvas.addEventListener('click', e => {
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

    function showAnnotationPopup(item, type, x, y) {
        const popup = document.getElementById('annotation-popup');
        const content = document.getElementById('popup-content');

        let html = '';
        if (type === 'correct') {
            html = `
                <div style="color:var(--success);font-weight:600;margin-bottom:6px;display:flex;align-items:center;gap:6px;">
                    <i data-lucide="check-circle" style="width:14px;height:14px;"></i> Correct Item
                </div>
                <div style="font-size:0.85rem;color:var(--text-muted);">${item.label || item.detected_label || 'Product'}</div>
                <div style="font-size:0.75rem;color:var(--text-dim);margin-top:4px;">Shelf ${item.shelf ?? item.detected_shelf ?? '?'}</div>
            `;
        } else if (type === 'misplaced') {
            const found = item.detected_label || item.label || 'Unknown Item';
            const expected = sanitizeMsg(item.expected_label);
            const hasExpected = expected && expected !== 'Unknown';
            if (hasExpected) {
                html = `
                    <div style="color:var(--danger);font-weight:600;margin-bottom:8px;display:flex;align-items:center;gap:6px;">
                        <i data-lucide="alert-triangle" style="width:14px;height:14px;"></i> Misplaced Product
                    </div>
                    <div style="font-size:0.84rem;margin-bottom:4px;"><span style="color:var(--success);font-weight:500;">Expected:</span> <span style="color:var(--text-muted);">${expected}</span></div>
                    <div style="font-size:0.84rem;margin-bottom:4px;"><span style="color:var(--danger);font-weight:500;">Found:</span> <span style="color:var(--text);">${found}</span></div>
                `;
            } else {
                html = `
                    <div style="color:var(--danger);font-weight:600;margin-bottom:8px;display:flex;align-items:center;gap:6px;">
                        <i data-lucide="alert-triangle" style="width:14px;height:14px;"></i> Unexpected Product
                    </div>
                    <div style="font-size:0.84rem;margin-bottom:4px;"><span style="color:var(--danger);font-weight:500;">Found:</span> <span style="color:var(--text);">${found}</span></div>
                    <div style="font-size:0.72rem;color:var(--text-dim);font-style:italic;margin-top:4px;">Not part of the reference image</div>
                `;
            }
        } else if (type === 'gap') {
            html = `
                <div style="color:var(--warning);font-weight:600;margin-bottom:8px;display:flex;align-items:center;gap:6px;">
                    <i data-lucide="square-dashed" style="width:14px;height:14px;"></i> Empty Space
                </div>
                <div style="font-size:0.84rem;color:var(--text-muted);">No product detected in this slot</div>
                <div style="font-size:0.75rem;color:var(--text-dim);margin-top:4px;">Shelf ${item.expected_shelf ?? '?'}</div>
                ${item.label && item.label !== 'Empty Space' ? `<div style="font-size:0.8rem;margin-top:4px;"><span style="color:var(--warning);">Space for:</span> ${item.label}</div>` : ''}
            `;
        }

        content.innerHTML = html;
        if (window.lucide) lucide.createIcons();

        const popupW = 300, popupH = 130;
        let left = x + 14;
        let top = y - 20;
        if (left + popupW > window.innerWidth) left = x - popupW - 14;
        if (top + popupH > window.innerHeight) top = window.innerHeight - popupH - 12;
        popup.style.left = left + 'px';
        popup.style.top = top + 'px';
        popup.style.display = 'block';

        const hidePopupOnScroll = () => {
            popup.style.display = 'none';
            window.removeEventListener('scroll', hidePopupOnScroll, true);
        };
        window.removeEventListener('scroll', hidePopupOnScroll, true);
        window.addEventListener('scroll', hidePopupOnScroll, true);
    }

    function displayResults(data) {
        scoreBadge.textContent = 'Analysis Complete';
        totalItems.textContent = data.total_items;
        const gapCount = data.gap_detections ? data.gap_detections.length : 0;
        const missingItemCount = data.missing_items ? data.missing_items.length : 0;
        const misplacedItemCount = data.misplaced_items ? data.misplaced_items.length : 0;

        missingItems.textContent = gapCount;
        misplacedItems.textContent = misplacedItemCount;

        issueList.innerHTML = '';
        if (gapCount === 0 && missingItemCount === 0 && misplacedItemCount === 0) {
            issueList.innerHTML = `<li class="empty-state" style="border-color: rgba(52, 211, 153, 0.4); color: var(--success); background: var(--success-soft);">
                <i data-lucide="check-circle"></i> Perfect shelf. No issues detected.
            </li>`;
            if (window.lucide) lucide.createIcons();
            displayComplianceImage(data);
            return;
        }

        if (data.missing_items && data.missing_items.length > 0) {
            data.missing_items.forEach(item => {
                const li = document.createElement('li');
                li.className = 'issue-item missing';
                li.innerHTML = `
                    <div style="flex:1;">
                        <strong style="display:block; margin-bottom:4px;">Product not found on shelf</strong>
                        <div style="font-size: 0.85rem; color: var(--text-muted);">
                            <span style="color: var(--warning); font-weight: 500;">Missing:</span> ${item.label}
                        </div>
                    </div>
                    <div style="text-align: right; color: var(--text-dim); font-size: 0.82rem;">
                        Shelf ${item.expected_shelf}
                    </div>
                `;
                issueList.appendChild(li);
            });
        }

        if (data.misplaced_items && data.misplaced_items.length > 0) {
            data.misplaced_items.forEach(item => {
                const li = document.createElement('li');
                li.className = 'issue-item misplaced';
                const expectedText = sanitizeMsg(item.expected_label) || 'Different product expected';
                const foundText = item.detected_label || item.label || 'Unknown Item';
                const hasExpected = expectedText && expectedText !== 'Unknown';
                li.innerHTML = `
                    <div style="flex:1;">
                        <strong style="display:block; margin-bottom:8px;">Wrong product in this slot</strong>
                        ${hasExpected ? `<div style="font-size: 0.84rem; padding: 5px 8px; border-radius: 6px; background: var(--success-soft); border-left: 2px solid var(--success); margin-bottom: 5px;">
                            <span style="color: var(--success); font-weight: 600;">Expected:</span> <span style="color: var(--text-muted);">${expectedText}</span>
                        </div>` : ''}
                        <div style="font-size: 0.84rem; padding: 5px 8px; border-radius: 6px; background: var(--danger-soft); border-left: 2px solid var(--danger);">
                            <span style="color: var(--danger); font-weight: 600;">Found:</span> <span style="color: var(--text); font-weight: 500;">${foundText}</span>
                        </div>
                    </div>
                `;
                issueList.appendChild(li);
            });
        }

        if (data.gap_detections && data.gap_detections.length > 0) {
            data.gap_detections.forEach(item => {
                const li = document.createElement('li');
                li.className = 'issue-item missing';
                const spaceLabel = item.label && item.label !== 'Empty Space' ? item.label : null;
                li.innerHTML = `
                    <div style="flex:1;">
                        <strong style="display:block; margin-bottom:4px;">Empty space. Product may be out of stock</strong>
                        ${spaceLabel
                            ? `<div style="font-size: 0.85rem; color: var(--text-muted);"><span style="color: var(--warning); font-weight: 500;">Expected here:</span> ${spaceLabel}</div>`
                            : `<div style="font-size: 0.85rem; color: var(--text-muted);">No product detected in this slot</div>`
                        }
                    </div>
                    <div style="text-align: right; color: var(--text-dim); font-size: 0.82rem;">
                        Shelf ${item.expected_shelf}
                    </div>
                `;
                issueList.appendChild(li);
            });
        }

        if (window.lucide) lucide.createIcons();
        displayComplianceImage(data);
    }
});
