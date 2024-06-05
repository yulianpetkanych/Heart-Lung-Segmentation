document.addEventListener("DOMContentLoaded", function() {
    var navbarHeight = document.querySelector('.header').offsetHeight;
    document.body.style.paddingTop = navbarHeight + 10 + 'px';
  });
  
  const dragAndDrop = (() => {
    const cardBody = document.querySelector('.card-body');
    const fileInput = document.getElementById('input-file');
    const browseButton = document.getElementById('browse-button');
    const buttons = document.querySelectorAll('.button-container button');
  
    const handleDragOver = (e) => {
        console.log("I'm over");
        e.preventDefault();
        cardBody.style.borderColor = '#333';
    };
  
    const handleDragLeave = (e) => {
        console.log("Goodbye");
        e.preventDefault();
        cardBody.removeAttribute('style');
    };
  
    const handleDrop = (e) => {
        console.log("files dropped");
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            if (validateFiles(files)) {
                activateButtons();
                const formData = new FormData();
                for (let i = 0; i < files.length; i++) {
                    formData.append('files', files[i]);
                }
                console.log(formData);
                uploadFiles(formData);
            } else {
                displayErrorMessage('Only .dcm files are allowed.');
            }
        }
    };
  
    const activateButtons = () => {
        buttons.forEach(button => {
            button.removeAttribute('disabled');
        });
    };
  
    const uploadFiles = (formData) => {
        fetch('http://127.0.0.1:8000/uploads', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(data);
            displaySuccessMessage('Files successfully uploaded.');
        })
        .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
            displayErrorMessage('Failed to upload files. Please try again.');
        });
    };
  
    const validateFiles = (files) => {
        for (let i = 0; i < files.length; i++) {
            if (!files[i].name.endsWith('.dcm')) {
                return false;
            }
        }
        return true;
    };
  
    cardBody.addEventListener('dragover', handleDragOver);
    cardBody.addEventListener('dragleave', handleDragLeave);
    cardBody.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', () => {
        const files = fileInput.files;
        if (files.length > 0) {
            if (validateFiles(files)) {
                activateButtons();
                const formData = new FormData();
                for (let i = 0; i < files.length; i++) {
                    formData.append('files', files[i]);
                }
                console.log(formData);
                uploadFiles(formData);
            } else {
                displayErrorMessage('Only .dcm files are allowed.');
                fileInput.value = ''; // Reset file input
            }
        }
    });
  
    browseButton.addEventListener('click', () => {
        fileInput.click();
    });
  })();
  
  let images = [];
  let currentIndex = 0;
  let isScrolling = false;
  
  function clearImages() {
    images.forEach(url => URL.revokeObjectURL(url));
    images = [];
    currentIndex = 0;
    document.getElementById('displayImage').src = '';
    updateProgressBar();
  }
  
  function sendButtonInfo(buttonId) {
      clearImages();
      const spinner = document.getElementById('loading-spinner');
      const displayImage = document.getElementById('displayImage');
      spinner.style.display = 'block';
      displayImage.style.display = 'none';
  
      fetch(`http://127.0.0.1:8000/get_images?button_id=${buttonId}`)
          .then(response => {
              if (!response.ok) {
                  throw new Error(`HTTP error! status: ${response.status}`);
              }
              return response.blob();
          })
          .then(blob => {
              const zip = new JSZip();
              return zip.loadAsync(blob);
          })
          .then(zip => {
              const imagePromises = [];
              Object.keys(zip.files).forEach(fileName => {
                  imagePromises.push(zip.files[fileName].async("blob").then(blob => {
                      const imageUrl = URL.createObjectURL(blob);
                      images.push(imageUrl);
                  }));
              });
              return Promise.all(imagePromises);
          })
          .then(() => {
              console.log(images.length);
              if (images.length > 0) {
                  currentIndex = 0;
                  displayImage.src = images[currentIndex];
                  displayImage.style.display = 'block';
                  updateProgressBar();
                  console.log("Image source set to:", displayImage.src); // Лог для перевірки
              }
              spinner.style.display = 'none';
          })
          .catch(error => {
              console.error('Error:', error);
              displayErrorMessage('Failed to load images. Please try again.');
              spinner.style.display = 'none';
          });
  }
  
  function prevImage() {
      if (images.length > 0) {
          currentIndex = (currentIndex - 1 + images.length) % images.length;
          document.getElementById('displayImage').src = images[currentIndex];
          updateProgressBar();
          console.log("Previous image source set to:", images[currentIndex]); // Лог для перевірки
      }
  }
  
  function nextImage() {
      if (images.length > 0) {
          currentIndex = (currentIndex + 1) % images.length;
          document.getElementById('displayImage').src = images[currentIndex];
          updateProgressBar();
          console.log("Next image source set to:", images[currentIndex]); // Лог для перевірки
      }
  }
  
  function updateProgressBar() {
      const progressBar = document.getElementById('progress-bar');
      const progressPercentage = ((currentIndex + 1) / images.length) * 100;
      progressBar.style.width = `${progressPercentage}%`;
      progressBar.textContent = `${Math.round(progressPercentage)}%`;
      console.log("Progress bar updated to:", progressPercentage); // Лог для перевірки
  }
  
  function handleWheel(event) {
      if (isScrolling) return;
  
      if (event.deltaY < 0) {
          prevImage();
      } else {
          nextImage();
      }
  
      isScrolling = true;
      setTimeout(() => {
          isScrolling = false;
      }, 200);
  }
  
  function displayErrorMessage(message) {
      const toast = document.getElementById('toast');
      const toastMessage = document.getElementById('toast-message');
      toastMessage.textContent = message;
      toast.className = "toast show";
      setTimeout(() => { toast.className = toast.className.replace("show", ""); }, 3000);
  }
  
  function displaySuccessMessage(message) {
      const toast = document.getElementById('toast');
      const toastMessage = document.getElementById('toast-message');
      toastMessage.textContent = message;
      toast.className = "toast show";
      setTimeout(() => { toast.className = toast.className.replace("show", ""); }, 3000);
  }
  
  document.getElementById('displayImage').addEventListener('wheel', event => {
      handleWheel(event);
      event.preventDefault();
  });
  
  // Додамо події для кнопок prev та next, щоб переконатися, що вони правильно визначені
  document.getElementById('prevButton').addEventListener('click', prevImage);
  document.getElementById('nextButton').addEventListener('click', nextImage);
  