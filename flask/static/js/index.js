const uploadButton = document.getElementById("uploadButton");
const imageUpload = document.getElementById("imageUpload");
uploadButton.addEventListener("click", function() {
    imageUpload.click();
    /*The click() method is a built-in JavaScript method that simulates a mouse click on an element.
     In this case, it simulates a click on the imageUpload input element.*/
 }); 

imageUpload.addEventListener("change", function() {
    const file = imageUpload.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        }
        reader.readAsDataURL(file);
    } else {
        imagePreview.style.display = 'none';
    }
});