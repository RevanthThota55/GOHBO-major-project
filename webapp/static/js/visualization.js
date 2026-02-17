/**
 * Visualization utilities for ExplainableMed-GOHBO
 * Handles charts, animations, and interactive visualizations
 */

/**
 * Create animated confidence bar
 */
function animateConfidenceBar(element, targetWidth, duration = 1000) {
    let start = null;
    const startWidth = 0;

    function step(timestamp) {
        if (!start) start = timestamp;
        const progress = Math.min((timestamp - start) / duration, 1);

        const currentWidth = startWidth + (targetWidth - startWidth) * easeOutQuart(progress);
        element.style.width = currentWidth + '%';

        if (progress < 1) {
            requestAnimationFrame(step);
        }
    }

    requestAnimationFrame(step);
}

/**
 * Easing function for smooth animations
 */
function easeOutQuart(x) {
    return 1 - Math.pow(1 - x, 4);
}

/**
 * Animate probability bars
 */
function animateProbabilityBars() {
    const bars = document.querySelectorAll('.probability-bar');

    bars.forEach((bar, index) => {
        const targetWidth = parseFloat(bar.style.width);
        bar.style.width = '0%';

        setTimeout(() => {
            animateConfidenceBar(bar, targetWidth, 800);
        }, index * 100);  // Stagger animation
    });
}

/**
 * Create smooth scroll to element
 */
function smoothScrollTo(element) {
    element.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

/**
 * Add fade-in animation to results
 */
function fadeInResults() {
    resultsSection.style.opacity = '0';
    resultsSection.style.display = 'block';

    let opacity = 0;
    const interval = setInterval(() => {
        if (opacity >= 1) {
            clearInterval(interval);
        }
        resultsSection.style.opacity = opacity;
        opacity += 0.05;
    }, 20);
}

/**
 * Highlight high-probability class
 */
function highlightTopPrediction(probabilities) {
    const items = document.querySelectorAll('.probability-item');
    let maxProb = 0;
    let maxIndex = 0;

    Object.values(probabilities).forEach((prob, index) => {
        if (prob > maxProb) {
            maxProb = prob;
            maxIndex = index;
        }
    });

    if (items[maxIndex]) {
        items[maxIndex].style.background = 'rgba(37, 99, 235, 0.05)';
        items[maxIndex].style.borderLeft = '4px solid var(--primary-blue)';
        items[maxIndex].style.paddingLeft = 'var(--spacing-sm)';
    }
}

/**
 * Create visual confidence indicator
 */
function createConfidenceIndicator(confidence) {
    const confidenceNum = parseFloat(confidence);

    let color, text, icon;

    if (confidenceNum >= 90) {
        color = 'var(--secondary-green)';
        text = 'High Confidence';
        icon = 'fa-check-circle';
    } else if (confidenceNum >= 75) {
        color = 'var(--accent-purple)';
        text = 'Good Confidence';
        icon = 'fa-check';
    } else {
        color = 'var(--accent-red)';
        text = 'Moderate Confidence';
        icon = 'fa-exclamation-triangle';
    }

    return {
        color: color,
        text: text,
        icon: icon
    };
}

/**
 * Add visual effects to images
 */
function addImageEffects() {
    const images = document.querySelectorAll('.image-box img');

    images.forEach(img => {
        img.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.05)';
            this.style.transition = 'transform 0.3s ease';
        });

        img.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
}

// Initialize animations when results are displayed
function initializeResultAnimations() {
    animateProbabilityBars();
    addImageEffects();
}