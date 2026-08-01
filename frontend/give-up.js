const finalTheme = document.getElementById('finalTheme');
finalTheme.volume = 0.85;

function playFinalTheme() {
  finalTheme.muted = false;
  finalTheme.play().catch(() => {});
}

playFinalTheme();
document.addEventListener('pointerdown', playFinalTheme);
document.addEventListener('touchstart', playFinalTheme);
document.addEventListener('keydown', event => {
  playFinalTheme();
  if (event.key === ' ') {
    event.preventDefault();
    window.location.href = '/landing.html';
  }
});
