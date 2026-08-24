document.addEventListener("DOMContentLoaded", function () {

  fetch("./json/01_banchan.json")
    .then(res => res.json())
    .then(data => {

      const box = document.querySelector(".banchan");

      data.forEach(item => {
        box.innerHTML += `
          <a href="#">
            <div class="product-card">
              <img src="${item.main_img}" alt="${item.name}">
              <h4>${item.name}</h4>
              <p>${item.description}</p>
              <h5>${item.price}원</h5>
            </div>
          </a>
        `;
      });

    });

});