/**
 * Movie Finder · NLP Movie Recommendation System
 * Frontend Controller for Natural Language Processing Engine
 */

document.addEventListener('DOMContentLoaded', () => {
    // State
    let allMovies = [];
    // Default selected IDs matching Image 1: Band of Brothers (id: 3) & The Godfather: Part II (id: 2)
    let selectedMovieIds = [3, 2];
    let selectedGenres = [];
    let currentRecommendations = [];
    let activeView = 'table'; // 'table' | 'netflix' | 'math'

    // DOM Elements
    const multiSelectBox = document.getElementById('multiSelectBox');
    const selectedChipsList = document.getElementById('selectedChipsList');
    const movieSearchInput = document.getElementById('movieSearchInput');
    const btnClearAll = document.getElementById('btnClearAll');
    const btnToggleDropdown = document.getElementById('btnToggleDropdown');
    const autocompleteDropdown = document.getElementById('autocompleteDropdown');
    const dropdownItemsList = document.getElementById('dropdownItemsList');
    const totalAvailableCount = document.getElementById('totalAvailableCount');

    // Filtering controls
    const filteringSection = document.getElementById('filteringSection');
    const accordionToggle = document.getElementById('accordionToggle');
    const genrePillGroup = document.getElementById('genrePillGroup');
    const ratingSlider = document.getElementById('ratingSlider');
    const ratingSliderVal = document.getElementById('ratingSliderVal');
    const yearMinInput = document.getElementById('yearMin');
    const yearMaxInput = document.getElementById('yearMax');
    const topNSelect = document.getElementById('topNSelect');
    const btnFindMovies = document.getElementById('btnFindMovies');

    // Results Views
    const tableViewContainer = document.getElementById('tableViewContainer');
    const netflixViewContainer = document.getElementById('netflixViewContainer');
    const mathViewContainer = document.getElementById('mathViewContainer');
    const tableBody = document.getElementById('tableBody');
    const cinemaCardsGrid = document.getElementById('cinemaCardsGrid');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const resultsCountNotice = document.getElementById('resultsCountNotice');
    const nlpDetailsContent = document.getElementById('nlpDetailsContent');

    // View Tabs
    const tabTableView = document.getElementById('tabTableView');
    const tabNetflixView = document.getElementById('tabNetflixView');
    const tabMathView = document.getElementById('tabMathView');

    // Modals
    const movieModal = document.getElementById('movieModal');
    const btnModalClose = document.getElementById('btnModalClose');
    const btnCloseModalSecondary = document.getElementById('btnCloseModalSecondary');
    const btnAddModalMovie = document.getElementById('btnAddModalMovie');
    const manageAppModal = document.getElementById('manageAppModal');
    const btnManageApp = document.getElementById('btnManageApp');
    const btnManageClose = document.getElementById('btnManageClose');
    const btnManageDone = document.getElementById('btnManageDone');

    let currentModalMovie = null;

    // 1. Initial Load: Fetch All Movies and Genres
    initApp();

    async function initApp() {
        try {
            // Load Movie Catalog
            const res = await fetch('/api/movies');
            allMovies = await res.json();
            totalAvailableCount.textContent = allMovies.length;
            document.getElementById('statsTotalMovies').textContent = allMovies.length;

            // Load Genres
            const genresRes = await fetch('/api/genres');
            const genres = await genresRes.json();
            renderGenrePills(genres);

            // Render pre-selected default chips
            renderChips();

            // Perform initial recommendation matching Image 1
            await fetchRecommendations();

        } catch (err) {
            console.error('Initialization error:', err);
        }
    }

    // 2. Multi-Select Chips Rendering
    function renderChips() {
        selectedChipsList.innerHTML = '';
        selectedMovieIds.forEach(id => {
            const movie = allMovies.find(m => m.id === id);
            if (!movie) return;

            const chip = document.createElement('div');
            chip.className = 'movie-chip';
            chip.innerHTML = `
                <span class="chip-title" title="${escapeHtml(movie.title)}">${escapeHtml(movie.title)}</span>
                <span class="chip-remove" data-id="${movie.id}">&times;</span>
            `;

            chip.querySelector('.chip-remove').addEventListener('click', (e) => {
                e.stopPropagation();
                removeMovieSelection(movie.id);
            });

            selectedChipsList.appendChild(chip);
        });
    }

    function addMovieSelection(id) {
        if (!selectedMovieIds.includes(id)) {
            selectedMovieIds.push(id);
            renderChips();
            movieSearchInput.value = '';
            closeDropdown();
            // Automatically re-recommend on new selection
            fetchRecommendations();
        }
    }

    function removeMovieSelection(id) {
        selectedMovieIds = selectedMovieIds.filter(mid => mid !== id);
        renderChips();
        fetchRecommendations();
    }

    btnClearAll.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedMovieIds = [];
        renderChips();
        fetchRecommendations();
    });

    // 3. Search and Autocomplete Dropdown
    function populateDropdown(filterText = '') {
        const query = filterText.toLowerCase().trim();
        dropdownItemsList.innerHTML = '';

        const filtered = allMovies.filter(m => {
            if (selectedMovieIds.includes(m.id)) return false; // Already selected
            if (!query) return true;
            return m.title.toLowerCase().includes(query) || m.genres.toLowerCase().includes(query);
        });

        if (filtered.length === 0) {
            dropdownItemsList.innerHTML = `
                <div style="padding: 14px; text-align: center; color: var(--text-dim); font-size: 13px;">
                    No matching movies found.
                </div>
            `;
            return;
        }

        filtered.slice(0, 30).forEach(movie => {
            const item = document.createElement('div');
            item.className = 'dropdown-item';
            item.innerHTML = `
                <span class="item-title">${escapeHtml(movie.title)}</span>
                <div class="item-meta">
                    <span class="item-genre">${movie.genres.split('|')[0]}</span>
                    <span>★ ${movie.avg_rating}%</span>
                </div>
            `;
            item.addEventListener('click', () => {
                addMovieSelection(movie.id);
            });
            dropdownItemsList.appendChild(item);
        });
    }

    function openDropdown() {
        multiSelectBox.classList.add('open');
        populateDropdown(movieSearchInput.value);
    }

    function closeDropdown() {
        multiSelectBox.classList.remove('open');
    }

    movieSearchInput.addEventListener('focus', openDropdown);
    movieSearchInput.addEventListener('input', () => {
        openDropdown();
        populateDropdown(movieSearchInput.value);
    });

    btnToggleDropdown.addEventListener('click', (e) => {
        e.stopPropagation();
        if (multiSelectBox.classList.contains('open')) {
            closeDropdown();
        } else {
            openDropdown();
            movieSearchInput.focus();
        }
    });

    document.addEventListener('click', (e) => {
        if (!multiSelectBox.contains(e.target)) {
            closeDropdown();
        }
    });

    // 4. Genre Pills Filtering
    function renderGenrePills(genres) {
        genres.forEach(g => {
            const pill = document.createElement('span');
            pill.className = 'genre-pill';
            pill.dataset.genre = g;
            pill.textContent = g;

            pill.addEventListener('click', () => {
                toggleGenre(g, pill);
            });
            genrePillGroup.appendChild(pill);
        });

        // "All Genres" pill handler
        const allPill = genrePillGroup.querySelector('[data-genre="all"]');
        allPill.addEventListener('click', () => {
            selectedGenres = [];
            genrePillGroup.querySelectorAll('.genre-pill').forEach(p => p.classList.remove('active'));
            allPill.classList.add('active');
            fetchRecommendations();
        });
    }

    function toggleGenre(genre, pillElement) {
        const allPill = genrePillGroup.querySelector('[data-genre="all"]');
        allPill.classList.remove('active');

        if (selectedGenres.includes(genre)) {
            selectedGenres = selectedGenres.filter(g => g !== genre);
            pillElement.classList.remove('active');
        } else {
            selectedGenres.push(genre);
            pillElement.classList.add('active');
        }

        if (selectedGenres.length === 0) {
            allPill.classList.add('active');
        }
        fetchRecommendations();
    }

    // 5. Accordion Toggle
    accordionToggle.addEventListener('click', () => {
        filteringSection.classList.toggle('active');
    });

    // Slider Listeners
    ratingSlider.addEventListener('input', () => {
        ratingSliderVal.textContent = ratingSlider.value + '%';
    });
    ratingSlider.addEventListener('change', fetchRecommendations);

    yearMinInput.addEventListener('change', fetchRecommendations);
    yearMaxInput.addEventListener('change', fetchRecommendations);
    topNSelect.addEventListener('change', fetchRecommendations);

    // Primary Button "Find similar Movies 🎬"
    btnFindMovies.addEventListener('click', fetchRecommendations);

    // Quick Presets Handler
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const preset = btn.dataset.preset;
            if (preset === 'mafia') {
                selectedMovieIds = [1, 17]; // The Godfather, Goodfellas
            } else if (preset === 'scifi') {
                selectedMovieIds = [15, 16]; // Inception, Interstellar
            } else if (preset === 'animation') {
                selectedMovieIds = [5, 9]; // Wallace & Gromit
            } else if (preset === 'netflix') {
                selectedMovieIds = [23, 26]; // Narcos, Stranger Things
            }
            renderChips();
            fetchRecommendations();
        });
    });

    // 6. Recommendation Fetch Pipeline
    async function fetchRecommendations() {
        loadingOverlay.classList.add('active');

        const payload = {
            movie_ids: selectedMovieIds,
            genres: selectedGenres,
            min_rating: parseFloat(ratingSlider.value),
            year_min: parseInt(yearMinInput.value) || 1900,
            year_max: parseInt(yearMaxInput.value) || 2030,
            top_n: parseInt(topNSelect.value) || 10
        };

        try {
            const res = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            currentRecommendations = data.recommendations || [];

            resultsCountNotice.textContent = `Showing Top ${currentRecommendations.length} Recommendations`;

            renderTable(currentRecommendations);
            renderCinemaGrid(currentRecommendations);
            renderMathInspector(currentRecommendations);

        } catch (err) {
            console.error('Recommendation fetch error:', err);
        } finally {
            loadingOverlay.classList.remove('active');
        }
    }

    // 7. Render Results Table (Image 1 Exact Layout)
    function renderTable(movies) {
        tableBody.innerHTML = '';

        if (movies.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="5" style="text-align: center; padding: 36px; color: var(--text-dim);">
                        No matching movies found for the selected filters. Try loosening your criteria or selecting more movies.
                    </td>
                </tr>
            `;
            return;
        }

        movies.forEach((m, idx) => {
            const tr = document.createElement('tr');

            // Construct row elements
            tr.innerHTML = `
                <td class="col-index">${m.rank || idx + 1}</td>
                <td class="col-title">
                    <div class="movie-title-cell">
                        <img src="${m.poster}" alt="Poster" class="mini-thumb" onerror="this.style.display='none'">
                        <span>${escapeHtml(m.title)}</span>
                    </div>
                </td>
                <td class="col-rating-you">
                    <div class="rating-bar-wrapper">
                        <div class="progress-track">
                            <div class="progress-fill" style="width: 0%" data-target-width="${m.score_for_you}%"></div>
                        </div>
                        <span class="rating-percent-text">${m.score_for_you}%</span>
                    </div>
                </td>
                <td class="col-rating-avg">${m.avg_rating}%</td>
                <td class="col-genres">${escapeHtml(m.genres)}</td>
            `;

            // Row click opens Netflix-style movie details modal
            tr.addEventListener('click', () => {
                openMovieModal(m);
            });

            tableBody.appendChild(tr);
        });

        // Trigger smooth progress bar width animation
        setTimeout(() => {
            document.querySelectorAll('.progress-fill').forEach(fill => {
                fill.style.width = fill.dataset.targetWidth;
            });
        }, 60);
    }

    // 8. Render Netflix-Style Cinema Poster Grid (Image 2 Aesthetic)
    function renderCinemaGrid(movies) {
        cinemaCardsGrid.innerHTML = '';

        movies.forEach(m => {
            const card = document.createElement('div');
            card.className = 'cinema-card';
            card.innerHTML = `
                <div class="cinema-poster-wrapper">
                    <img src="${m.poster}" alt="${escapeHtml(m.title)}" loading="lazy">
                    <span class="card-match-badge">${m.score_for_you}% Match</span>
                </div>
                <div class="cinema-card-info">
                    <div class="card-movie-title">${escapeHtml(m.title)}</div>
                    <div class="card-genres">${escapeHtml(m.genres.replace(/\|/g, ' • '))}</div>
                    <div class="card-footer-meta">
                        <span>${m.year}</span>
                        <span class="card-avg-score">★ ${m.avg_rating}%</span>
                    </div>
                </div>
            `;

            card.addEventListener('click', () => {
                openMovieModal(m);
            });

            cinemaCardsGrid.appendChild(card);
        });
    }

    // 9. Render NLP Vector Space Inspector
    function renderMathInspector(movies) {
        if (!movies.length) {
            nlpDetailsContent.innerHTML = `<p>No recommendations to inspect.</p>`;
            return;
        }

        const topMovie = movies[0];
        const sharedTerms = topMovie.top_shared_terms || [];

        let termsHtml = sharedTerms.map(t => `
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #232736;">
                <span style="color: #60a5fa; font-family: var(--font-mono); font-weight: 600;">"${escapeHtml(t.term)}"</span>
                <span style="color: var(--primary); font-family: var(--font-mono); font-weight: 700;">+${t.importance} importance weight</span>
            </div>
        `).join('');

        if (!termsHtml) {
            termsHtml = `<p style="color: var(--text-dim);">Cosine dot product computed across 1,200+ global vocabulary vector dimensions.</p>`;
        }

        nlpDetailsContent.innerHTML = `
            <div style="margin-bottom: 14px;">
                <h4 style="color: #ffffff; margin-bottom: 6px;">
                    NLP Breakdown for #1 Recommended: <span style="color: var(--primary);">${escapeHtml(topMovie.title)}</span>
                </h4>
                <p style="color: var(--text-muted); font-size: 13px;">
                    Cosine Similarity Score: <strong>${topMovie.similarity_raw}</strong> (Normalized Rating: <strong>${topMovie.score_for_you}%</strong>)
                </p>
            </div>
            <div style="background: #10121a; border: 1px solid var(--border-color); border-radius: var(--radius-sm); padding: 12px;">
                <h5 style="color: var(--text-dim); font-size: 11px; text-transform: uppercase; margin-bottom: 8px;">Top Shared NLP Tokens Contributing to Match</h5>
                ${termsHtml}
            </div>
        `;
    }

    // 10. View Mode Switcher
    tabTableView.addEventListener('click', () => setView('table'));
    tabNetflixView.addEventListener('click', () => setView('netflix'));
    tabMathView.addEventListener('click', () => setView('math'));

    function setView(view) {
        activeView = view;
        [tabTableView, tabNetflixView, tabMathView].forEach(t => t.classList.remove('active'));
        [tableViewContainer, netflixViewContainer, mathViewContainer].forEach(c => c.classList.add('hidden'));

        if (view === 'table') {
            tabTableView.classList.add('active');
            tableViewContainer.classList.remove('hidden');
        } else if (view === 'netflix') {
            tabNetflixView.classList.add('active');
            netflixViewContainer.classList.remove('hidden');
        } else if (view === 'math') {
            tabMathView.classList.add('active');
            mathViewContainer.classList.remove('hidden');
        }
    }

    // 11. Movie Modal (Cinematic Backdrop & NLP Insights)
    function openMovieModal(movie) {
        currentModalMovie = movie;

        document.getElementById('modalMovieTitle').textContent = movie.title;
        document.getElementById('modalRatingBadge').textContent = `${movie.score_for_you || 90}% Match for You`;
        document.getElementById('modalYear').textContent = movie.year;
        document.getElementById('modalGenres').textContent = movie.genres.replace(/\|/g, ', ');
        document.getElementById('modalAvgScore').textContent = `${movie.avg_rating}%`;
        document.getElementById('modalPosterImg').src = movie.poster;
        document.getElementById('modalOverviewText').textContent = movie.overview;
        document.getElementById('modalDirector').textContent = movie.director || 'N/A';
        document.getElementById('modalCast').textContent = Array.isArray(movie.cast) ? movie.cast.join(', ') : (movie.cast || 'N/A');

        // Backdrop
        const backdropUrl = movie.backdrop || movie.poster;
        document.getElementById('modalHero').style.backgroundImage = `url('${backdropUrl}')`;

        // Shared NLP tags
        const tagContainer = document.getElementById('modalNlpTags');
        tagContainer.innerHTML = '';
        const tags = movie.shared_keywords || (movie.keywords || []).slice(0, 6);
        tags.forEach(tag => {
            const tagEl = document.createElement('span');
            tagEl.className = 'nlp-tag';
            tagEl.textContent = `#${tag}`;
            tagContainer.appendChild(tagEl);
        });

        movieModal.classList.add('active');
    }

    function closeMovieModal() {
        movieModal.classList.remove('active');
    }

    btnModalClose.addEventListener('click', closeMovieModal);
    btnCloseModalSecondary.addEventListener('click', closeMovieModal);
    movieModal.addEventListener('click', (e) => {
        if (e.target === movieModal) closeMovieModal();
    });

    btnAddModalMovie.addEventListener('click', () => {
        if (currentModalMovie) {
            addMovieSelection(currentModalMovie.id);
            closeMovieModal();
        }
    });

    // 12. Manage App Modal
    btnManageApp.addEventListener('click', () => {
        manageAppModal.classList.add('active');
    });
    btnManageClose.addEventListener('click', () => manageAppModal.classList.remove('active'));
    btnManageDone.addEventListener('click', () => manageAppModal.classList.remove('active'));
    manageAppModal.addEventListener('click', (e) => {
        if (e.target === manageAppModal) manageAppModal.classList.remove('active');
    });

    // Utility
    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});
