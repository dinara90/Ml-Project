const video = document.getElementById('video');
const button = document.getElementById('capture');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(error => {
    console.error("Ошибка при доступе к камере:", error);
  });

function captureImage() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  const dataURL = canvas.toDataURL('image/jpeg');
  console.log('We are expecting ohoy');
  sendImageToServer(dataURL);
}

button.addEventListener('click', captureImage);


function sendImageToServer(base64Image) {
  const url = 'http://localhost:8000/predict';
  const body = JSON.stringify({ image: base64Image });

  fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: body
  })
  .then(response => response.json())
  .then(data => {
    console.log("Предсказание:", data);
    if (data === 'one') {
      fetch('http://localhost:8000/send', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify("one")
      })
      .then(response => response.json())
      .then(data => console.log(data))
      .catch(error => console.error('Ошибка:', error));
    }
    
    displayResults(data);
  })
  .catch(error => {
    console.error("Ошибка при отправке изображения:", error);
  });
}

function displayResults(data) {
  console.log("Отображение результатов:", data);
  const resultsElement = document.getElementById('results');
  resultsElement.innerHTML = '';

  const predictionElement = document.createElement('div');
  predictionElement.classList.add('prediction');
  predictionElement.textContent = `Предсказание: ${data}`;
  resultsElement.appendChild(predictionElement);
}
