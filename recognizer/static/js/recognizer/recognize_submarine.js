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
    console.log(data);
    var layout = {
        title: 'Probability distribution between classes',
        titlefont: {
            family: defaultFont,
            size: 20,
            color: 'black'
        },
        xaxis: {
            title: 'Classes: submarine`s depth ',
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
    window.barScores = await Plotly.newPlot('bar-scores', [data], layout);
    revealElement("bar-scores");
}

function revealElement(elementId) {
    let element = document.getElementById(elementId);
    element.setAttribute("class", element.getAttribute("class").concat(" reveal-down"));
}

async function showSound(stats) {
    var sound = {
        x: stats["freqs"],
        y: stats["amps"],
        type: "scatter"
    };
    var layout = {
        title: 'Parameters obtained by using FFT',
        xaxis: {
            title: 'Frequency (Hz)',
            titlefont: {
                family: defaultFont,
                size: 18,
                color: '#7f7f7f'
            }
        },
        yaxis: {
            title: 'Normalized amplitude',
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

let fixedSize = 3    //Fix size of float after num 
function showStats(stats) {
    let statsHolder = document.getElementById("recognize-info");
    statsHolder.appendChild(createCollectionItem(`Neural network name: ${stats["nn_name"]}`));
    console.log(stats);
    statsHolder.appendChild(createCollectionItem(`Features shape: ( ${stats["features_shape"].map(el => el == null ? "1" : el).join(",")} )`));
    statsHolder.appendChild(createCollectionItem(`Recognized class: ${stats["class"]}`));
    statsHolder.appendChild(createCollectionItem(`Score: ${(stats["score"]*100).toFixed(fixedSize)} %`));
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

// TODO: Refactor this shit! :)
async function saveToPdf() {
    let y_shift = 20;
    let doc = new jsPDF('p', 'pt', 'a4');
    doc.setFontSize(20);
    doc.text(20, y_shift, 'Submarine s depth recognition');
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

    let leftPadding = 10;
    doc.addImage(mfccPlot, "JPEG", leftPadding, y_shift, 500, 300);
    y_shift += 300 + 20;

    let statsHolder = document.getElementById("recognize-info");

    Array.from(statsHolder.getElementsByTagName("li"))
        .forEach(el => {
            doc.text(leftPadding, y_shift, el.innerHTML);
            y_shift += 20;
        })
    // doc.addHTML(stats);
    doc.addImage(barPlot, "JPEG", leftPadding, y_shift + 10, 500, 300);
    //TODO: Show the date and user:
    doc.save("submarine_report.pdf");
}