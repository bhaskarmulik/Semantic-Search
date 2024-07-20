function isElementInViewport(el) {
    const rect = el.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

function handleScroll() {
    const elements = document.querySelectorAll('.fade-in');
    elements.forEach(element => {
        if (isElementInViewport(element)) {
            element.classList.add('visible');
        }
    });
}

function getTextInput() {
    let input = document.getElementById('query').value;
    fetch('/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: input })
    });
    window.location.href = '/results';
}

window.addEventListener('scroll', handleScroll);
window.addEventListener('resize', handleScroll);
window.addEventListener('load', handleScroll);