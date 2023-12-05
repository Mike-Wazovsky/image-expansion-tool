const grabBtn = document.getElementById("grabBtn");

let fileArrayBuffer = "file Array Buffer DiDn'T CHANGE!!"

grabBtn.addEventListener("click",() => {
    // Получить активную вкладку браузера
    chrome.tabs.query({active: true}, function(tabs) {
        var tab = tabs[0];
        // и если она есть, то выполнить на ней скрипт
        if (tab) {
            execScript(tab);
        } else {
            alert("There are no active tabs")
        }
    })
})

/**
 * Выполняет функцию grabImages() на веб-странице указанной
 * вкладки и во всех ее фреймах,
 * @param tab {Tab} Объект вкладки браузера
 */
function execScript(tab) {
    // Выполнить функцию на странице указанной вкладки
    // и передать результат ее выполнения в функцию onResult
    chrome.scripting.executeScript(
        {
            target:{tabId: tab.id, allFrames: true},
            func:grabImages
        },
        onResult
    )
}

/**
 * Функция исполняется на удаленной странице браузера,
 * получает список изображений и возвращает массив
 * путей к ним
 *
 *  @return Array массив строк
 */
function grabImages() {
    const images = document.querySelectorAll("img");
    return Array.from(images).map(image=>image.src);
}

/**
 * Выполняется после того как вызовы grabImages
 * выполнены во всех фреймах удаленной web-страницы.
 * Функция объединяет результаты в строку и копирует
 * список путей к изображениям в буфер обмена
 *
 * @param {[]InjectionResult} frames Массив результатов
 * функции grabImages
 */
function onResult(frames) {
    // Если результатов нет
    if (!frames || !frames.length) {
        alert("Could not retrieve images from specified page");
        return;
    }
    // Объединить списки URL из каждого фрейма в один массив
    const imageUrls = frames.map(frame=>frame.result)
        .reduce((r1,r2)=>r1.concat(r2));

    imageRequest()

    // downloadImage(imageUrls, outputPath)

    // Скопировать в буфер обмена полученный массив
    // объединив его в строку, используя символ перевода строки
    // как разделитель
    window.navigator.clipboard
        .writeText(imageUrls.join("\n"))
        .then(()=>{
            // закрыть окно расширения после завершения
            window.close();
        });
}

async function serverCheck(imageUrl) {
    const response = fetch('http://18.215.145.127:12023/ping')
        .then(response => response)
        .then(data => console.log(data))
        .catch(error => console.error('Request error:', error));

    alert(response)
}
async function imageRequest() {

    const response = await chrome.runtime.sendMessage({ message: "save_text" })
    // TODO(fix response receive)
    if (response.success) {
        alert(response.res)
        const url = URL.createObjectURL(response.res);

        const a = document.createElement('a');
        a.href = url;
        a.download = 'prediction.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    } else {
        // Обрабатываем ошибку, если запрос не удался
        console.error('Error:', response.error);
    }
}