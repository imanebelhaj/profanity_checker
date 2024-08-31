
// document.getElementById('insultForm').onsubmit = function(event) {
//     event.preventDefault();
//     const text = document.getElementById('text').value;
//     const lang = document.getElementById('lang').value;

//     const formData = new URLSearchParams();
//     formData.append('text', text);
//     formData.append('lang', lang);

//     fetch('/predict', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/x-www-form-urlencoded'
//         },
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         document.getElementById('result').textContent = data.result;
//     })
//     .catch(error => console.error('Error:', error));
// };


document.getElementById('insultForm').onsubmit = function(event) {
    event.preventDefault();
    const text = document.getElementById('text').value;
    const lang = document.getElementById('lang').value;

    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text, lang: lang })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').textContent = data.result;
    })
    .catch(error => console.error('Error:', error));
};
