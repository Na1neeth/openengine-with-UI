// ===========================================================================
// OpenEngine Web UI — Client-side JavaScript
// ===========================================================================

(function () {
    'use strict';

    // ---- Theme Toggle ----
    const themeToggle = document.getElementById('themeToggle');
    const moonIcon = document.getElementById('moonIcon');
    const sunIcon = document.getElementById('sunIcon');
    const html = document.documentElement;

    function setTheme(theme) {
        html.setAttribute('data-theme', theme);
        localStorage.setItem('openengine-theme', theme);
        if (theme === 'dark') {
            moonIcon.style.display = 'block';
            sunIcon.style.display = 'none';
        } else {
            moonIcon.style.display = 'none';
            sunIcon.style.display = 'block';
        }
    }

    // Initialize theme
    const savedTheme = localStorage.getItem('openengine-theme') || 'dark';
    setTheme(savedTheme);

    if (themeToggle) {
        themeToggle.addEventListener('click', function () {
            const current = html.getAttribute('data-theme');
            setTheme(current === 'dark' ? 'light' : 'dark');
        });
    }

    // ---- Mobile Navigation ----
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const mobileNavOverlay = document.getElementById('mobileNavOverlay');
    const mobileNavClose = document.getElementById('mobileNavClose');

    if (mobileMenuBtn && mobileNavOverlay) {
        mobileMenuBtn.addEventListener('click', function () {
            mobileNavOverlay.classList.add('open');
        });

        mobileNavClose.addEventListener('click', function () {
            mobileNavOverlay.classList.remove('open');
        });

        mobileNavOverlay.addEventListener('click', function (e) {
            if (e.target === mobileNavOverlay) {
                mobileNavOverlay.classList.remove('open');
            }
        });
    }

    // ---- Flash Messages Auto-Dismiss ----
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(function (msg) {
        setTimeout(function () {
            msg.style.transition = 'opacity 300ms ease, transform 300ms ease';
            msg.style.opacity = '0';
            msg.style.transform = 'translateY(-8px)';
            setTimeout(function () { msg.remove(); }, 300);
        }, 6000);
    });

})();
