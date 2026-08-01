const landingTheme = document.getElementById('landingTheme');
landingTheme.volume = 0.85;

function playLandingTheme() {
  landingTheme.muted = false;
  landingTheme.play().catch(() => {});
}

playLandingTheme();
document.addEventListener('pointerdown', playLandingTheme);
document.addEventListener('touchstart', playLandingTheme);
document.addEventListener('keydown', event => {
  playLandingTheme();
  if (event.key === ' ') {
    event.preventDefault();
    window.location.href = '/countdown.html';
  }
});
