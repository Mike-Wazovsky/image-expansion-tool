chrome.runtime.onMessage.addListener(
    async (request, sender, sendResponse) => {
        switch (request.message) {
            case 'save_text': {
                const serverURL = 'http://18.215.145.127:12023/predict';

                const requestData = {
                    images: [
                        [
                            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.4, 0.5, 0.6]],
                            [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2], [0.4, 0.5, 0.6]],
                            [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2], [0.4, 0.5, 0.6]]
                        ],
                        // Добавьте еще вложенных массивов, если необходимо
                    ]
                };

                const requestOptions = {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestData)
                };

                const response = await fetch(serverURL, requestOptions)
                    .then(response => response.blob())
                    .then(blob => {
                        // Создание объекта URL для создания ссылки на файл
                        sendResponse({success: true, res: blob});
                    })
                    .catch(error => console.error('Error:', error));

                return true;
            }
            default: {
                sendResponse({success: true, res: "Default request"});
                break;
            }
        }
    }
);

async function makeHTTPRequest(serverURL, requestOptions) {
    const response = await fetch(serverURL, requestOptions)
    fileArrayBuffer = response.blob()
    return response
}