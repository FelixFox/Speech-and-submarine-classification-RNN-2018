class ProgressBar {
    constructor(elementId) {
        this.bar = document.getElementById(elementId);
    }

    activate() {
        this.bar.style = "display: block;";
    }

    deactivate() {
        this.bar.style = "display: none;";
    }
}
let progressBar = new ProgressBar("progress-bar")
let defaultFont = `apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Oxygen-Sans,Ubuntu,Cantarell,"Helvetica Neue",sans-serif`;

document.getElementById("file-upload-holder").onchange = async () => {
    console.log("Sent file!");

    var formData = new FormData(document.forms.sound)

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "", true);
    xhr.send(formData);
    progressBar.activate()
    xhr.onload = async () => {
        progressBar.deactivate();
        if (xhr.status != 200) {
            console.log(xhr.statusText)
        } else {
            var response = JSON.parse(xhr.response);
            showRecognizedStats(response)
            window.recognizeStats = response;
        }
    }
};

async function showRecognizedStats(stats) {
    document.getElementById("recognize-result").style = "display: block;";

    await showSound(stats);
    showStats(stats);
    await showPredictionsHistogram(stats);
}

async function delayedScroll(time, fn) {
    await delay(time);
    fn();
    window.scroll({
        top: window.innerHeight,
        left: 0,
        behavior: 'smooth'
    });
}

async function delay(time) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve();
        }, time)
    })
}

async function showPredictionsHistogram(stats) {
    var data = {
        x: stats["all_scores"].map(scoreItem => scoreItem["class"]),
        y: stats["all_scores"].map(scoreItem => scoreItem["score"]),
        type: "bar"
    };
    var layout = {
        title: 'Probability distribution between classes',
        titlefont: {
            family: defaultFont,
            size: 20,
            color: 'black'
        },
        xaxis: {
            title: 'Classes: words',
            titlefont: {
                family: defaultFont,
                size: 18,
                color: '#7f7f7f'
            }
        },
        yaxis: {
            title: 'Probability',
            titlefont: {
                family: defaultFont,
                size: 18,
                color: '#7f7f7f'
            }
        }
    };
    console.log(data);
    window.barScores = await Plotly.newPlot('bar-scores', [data], layout);
    revealElement("bar-scores");
}

function revealElement(elementId) {
    let element = document.getElementById(elementId);
    element.setAttribute("class", element.getAttribute("class").concat(" reveal-down"));
}

async function showSound(stats) {
    var sound = {
        z: stats["mels"],

        type: "heatmap"
    };
    var layout = {
        title: 'Mel-frequency cepstral coefficients',
        titleFont: defaultFont,
        xaxis: {
            title: 'Time step (1 step = ~15 ms)',
            titlefont: {
                family: defaultFont,
                size: 18,
                color: '#7f7f7f'
            }
        },
        yaxis: {
            title: 'Amplitude (db)',
            titlefont: {
                family: defaultFont,
                size: 18,
                color: '#7f7f7f'
            }
        }
    };
    window.soundPlot = await Plotly.newPlot('plot', [sound], layout);
    revealElement("plot");
}

function createCollectionItem(text) {
    var li = document.createElement("li");
    li.setAttribute("class", "collection-item reveal-down")
    li.appendChild(document.createTextNode(text));
    return li;
}

let fixedSize = 3
function showStats(stats) {
    let statsHolder = document.getElementById("recognize-info");
    statsHolder.appendChild(createCollectionItem(`Neural network name: ${stats["nn_name"]}`));
    statsHolder.appendChild(createCollectionItem(`Features shape:  (${stats["features_shape"].map(el => el == null ? "1" : el).join(",")} )`));
    statsHolder.appendChild(createCollectionItem(`Recognized class: ${stats["class"]}`));
    statsHolder.appendChild(createCollectionItem(`Score: ${(stats["score"] * 100).toFixed(fixedSize)} %`));
    statsHolder.appendChild(createCollectionItem(`Sample rate: ${stats["sample_rate"]}`));
    statsHolder.appendChild(createCollectionItem(`Recognition duration: ${stats["time_recognition"]}`));
}

function getCurrentDate() {
    let date = new Date();
    let day = date.getDate();
    let month = date.getMonth() + 1;
    let year = date.getFullYear();
    let hours = date.getHours();
    let minutes = date.getMinutes();
    let seconds = date.getSeconds();
    return `${day}/${month}/${year}   ${hours}:${minutes}:${seconds}`;


}
async function saveToPdf() {
    let doc = new jsPDF('p', 'pt', 'a4');
    let y_shift = 20;
    doc.setFontSize(20);
    doc.text(20, y_shift, 'Speech recognition');
    y_shift += 30;
    doc.setFontSize(15);
    //console.log(window.username);
    doc.text(20, y_shift, 'Username:   ' + window.username);
    y_shift += 20;
    doc.text(20, y_shift, 'Date:   ' + getCurrentDate());
    y_shift += 20;

    doc.setFontSize(14);
    let mfccPlot = await Plotly.toImage(window.soundPlot, { height: 600, width: 1200 });
    let barPlot = await Plotly.toImage(window.barScores, { height: 600, width: 1200 });
    console.log(window.soundPlot)
    let leftPadding = 10;
    doc.addImage(mfccPlot, "JPEG", leftPadding, y_shift, 500, 300);

    let statsHolder = document.getElementById("recognize-info");

    y_shift += 20 + 300;
    Array.from(statsHolder.getElementsByTagName("li"))
        .forEach(el => {
            doc.text(leftPadding, y_shift, el.innerHTML);
            y_shift += 20;
        })
    // doc.addHTML(stats);
    doc.addImage(barPlot, "JPEG", leftPadding, y_shift + 10, 500, 300);
    doc.save("speech_report.pdf");
}

// $(document).ready(function () {
//     var getImageFromUrl = function (url, callback) {
//         var img = new Image();
//         img.onError = function () {
//             alert('Cannot load image: "' + url + '"');
//         };
//         img.onload = function () {
//             callback(img);
//         };
//         img.src = url;
//     }
//     var createPDF = function (imgData) {
//         var doc = new jsPDF('p', 'pt', 'a4');
//         var width = doc.internal.pageSize.width;
//         var height = doc.internal.pageSize.height;
//         var options = {
//             pagesplit: true
//         };
//         doc.text(10, 20, 'Crazy Monkey');
//         var h1 = 50;
//         var aspectwidth1 = (height - h1) * (9 / 16);
//         doc.addImage(imgData, 'JPEG', 10, h1, aspectwidth1, (height - h1), 'monkey');
//         doc.addPage();
//         doc.text(10, 20, 'Hello World');
//         var h2 = 30;
//         var aspectwidth2 = (height - h2) * (9 / 16);
//         doc.addImage(imgData, 'JPEG', 10, h2, aspectwidth2, (height - h2), 'monkey');
//         doc.output('datauri');
//     }
//     getImageFromUrl('thinking-monkey.jpg', createPDF);
// });