export default class mobilenetv2Classification {
    constructor() {    
        this.label = ['Amazon',
            'American Bank',
             'Axis Bank',
             'Bandhan Bank',
             'bigbasket',
             'Bank of America',
             'Bank of Baroda',
             'Bank of India',
             'Bank of Maharashtra',
             'Bse',
             'Canara Bank',
             'Central Bank of India',
             'cdsl',
             'Citibank',
             'CitibankUS',
             'Dropbox',
             'etrade',
             'Facebook',
             'Federal Bank',
             'Flipkart',
             'Google',
             'groww',
             'HDFC Bank',
             'hsbc',
             'Indian Bank',
             'ICICI Bank',
             'IDBI Bank',
             'idfc',
             'ikea',
             'incometax',
             'IndusInd Bank',
             'Indian Overseas Bank',
             'JP Morgan',
             'Kotak Bank',
             'LinkedIn',
             'NSDL',
             'Ola',
             'Paypal',
             'Paytm',
             'Pinterest',
             'Punjab National Bank',
             'Punjab and Sind Bank',
             'RBL Bank',
             'State Bank of India',
             'Swiggy',
             'Twitter',
             'Uber',
             'UCO Bank',
             'Union Bank of India',
             'Upstox',
             'US Bank',
             'Whatsapp',
             'Yes Bank',
             'Zerodha',
             'Zomato'];
        this.count = 0;
	this.threshold = 0.85;
        this.imageWidth = 224;
        this.imageHeight = 224;
        this.width = 400;
        this.height = 400;
    }

    drawCorrespondence(x_origin, y_origin, outputs, ans, image) {
        return new Promise((resolve, reject) => {
            var canvas = document.createElement("canvas");
            canvas.width = image.width;
            canvas.height = image.height;
            var ctx = canvas.getContext("2d");
            ctx.drawImage(image, x_origin, y_origin, image.width, image.height);
            resolve(canvas.toDataURL("image/png"))
        });
    }

    displayPrediction(outputs, image, debugFlag) {
        return new Promise( (resolve, reject) => {
            let ans = -1;
            console.log(outputs);

            console.log("Top 3 Predictions:-")

            //Make clone of original array
            let tmp_outputs = [...outputs];
            tmp_outputs.sort((a, b) => b-a);
            for(let i=0; i<3; ++i){
                console.log(i+1 + ". Class: " + this.label[outputs.indexOf(tmp_outputs[i])] + ", Confidence: "+ (tmp_outputs[i]*100).toFixed(2))
            }

            let max_confidence = Math.max(...outputs);
            let max_confidence_index = outputs.indexOf(max_confidence);

            console.log("Predicted Class: " + this.label[max_confidence_index] + ",Max Confidence: "+ (max_confidence*100).toFixed(2));

            if (max_confidence > this.threshold){
                ans = max_confidence;
            }
            if (ans == -1){
                resolve(null);
            }
            this.drawCorrespondence(0, 0, outputs, ans, image)
                .then((corr_image) => resolve({
                    type : 'custom',
                    site : this.label[max_confidence_index],
                    confidence : max_confidence * 100,
                    corr_img : corr_image,
                    display_confidence: ''
                }));
        });
    }

    async predict(url, graph_model_url) {
        // Start TF scope
        tf.engine().startScope()
        this.model = await tf.loadGraphModel(graph_model_url)

        console.log('I am inside tfJS');
        let image = await this.loadImage(url);

        let t0 = performance.now();
        
        let tf_image = await tf.browser.fromPixels(image);

        // Cast  int32 to float32 
        tf_image = tf.cast(tf_image, 'float32')
    
    
        /*const smallImg = await tf.image.resizeBilinear(tf_image, [224, 224]);
        tf_image = smallImg*/

        let tf_img = await tf_image.expandDims(0);
        console.log("tf_image shape: ", tf_img.shape);

        let t1 = performance.now();
        console.log("Image to tensor took " + (t1 - t0).toFixed(2) + " milliseconds.");

        let backend = 'webgl';
        await tf.setBackend(backend);
        console.log('tf backend ' + backend + ' configured');

        let t01 = performance.now();
        try {

            //let outputs = await this.model.executeAsync(tf_img)
            let outputs = await this.model.predict(tf_img);
            outputs = await outputs.data();
            let t11 = performance.now();
            console.log("Prediction took " + (t11 - t01).toFixed(2) + " milliseconds.");

            let pred_result = await this.displayPrediction(outputs, image);
            this.count = this.count + 1;
            console.log(pred_result);

            if (pred_result != null) {
                return {
                    site: pred_result.site,
                    confidence: pred_result.confidence,
                    time_taken: (Math.round((t11 - t01) / 1000)),
                    image: pred_result.corr_img
                };
            } else {
                return {
                    site: 'NaN',
                    confidence: 0,
                    time_taken: Math.round((t11 - t01) / 1000),
                    image: url
                };
            }
        } catch (error) {
            console.log("h3", error);
            let t11 = performance.now();
            console.log("Prediction took " + (t11 - t01).toFixed(2) + " milliseconds.");
            return {
                site: 'NaN',
                confidence: 0,
                time_taken: Math.round((t11 - t01) / 1000),
                image: url
            };
        } finally {
            console.log("Finally Block")

            //Free Variables
            tf_image.dispose()  
            tf_img.dispose()
            this.model.dispose()

            console.log("Memory stats", tf.memory().numTensors,  Math.round(tf.memory().numBytes/(1024*1024)), "MB");
            console.log("GPU stats", Math.round(tf.memory().numBytesInGPU/(1024*1024), 2), "MB");
            console.log("finally")
            tf.engine().endScope();     
        }
    }

    loadImage(imageUrl) {
            // * Loads the image by making a HTML Image Element from the obtaied B64
        return new Promise((resolve, reject) => {
            let img = new Image(this.imageWidth, this.imageHeight);
            img.crossOrigin = '';
            img.onload = () => {
                resolve(img);
            };
            img.onerror = () => {
                reject("Error: Unable to load image");
            };
            img.src = imageUrl;
        });
    }

    //----------------------------------------------------------------------------------
    //---------------------------------------functions for debug mode-------------------
    debugDrawCorrespondence(x_origin, y_origin, outputs, ans, image) {
        return new Promise((resolve, reject) => {
            var canvas = document.createElement("canvas");
            canvas.width = image.width;
            canvas.height = image.height;
            var ctx = canvas.getContext("2d");
            ctx.drawImage(image, x_origin, y_origin, image.width, image.height);
            let y_min = outputs[0].dataSync()[4 * ans] * image.height;
            let x_min = outputs[0].dataSync()[4 * ans + 1] * image.width;
            let y_max = outputs[0].dataSync()[4 * ans + 2] * image.height;
            let x_max = outputs[0].dataSync()[4 * ans + 3] * image.width;
            ctx.beginPath();
            ctx.lineWidth = 3;
            ctx.strokeStyle = "red";
            ctx.rect(x_min + x_origin, y_origin + y_min, x_max - x_min, y_max - y_min);
            ctx.stroke();
            ctx.lineWidth = 1;
            ctx.font = "normal 15px Georgia";
            ctx.fillText(this.label[outputs[2].dataSync()[ans] - 1] + ':' + Math.round(outputs[1].dataSync()[ans] * 100) + '%', x_min + x_origin, y_max + 15 + y_origin);
            ctx.strokeText(this.label[outputs[2].dataSync()[ans] - 1] + ':' + Math.round(outputs[1].dataSync()[ans] * 100) + '%', x_min + x_origin, y_max + 15 + y_origin);
            resolve(canvas.toDataURL("image/png"))
        });
    }

    debugDisplayPrediction(outputs, image) {
        return new Promise((resolve, reject) => {
            let ans = -1;
            for (let i = 0; i < outputs[1].dataSync().length; i++) {
                if (outputs[1].dataSync()[i] > 0.3) {
                    ans = i;
                    break;
                }
            }
            if (ans == -1) {
                console.log('No prediction from tfMLPrediction');
                resolve(null)
            }
            console.log(this.label[outputs[2].dataSync()[ans] - 1] + ':' + Math.round(outputs[1].dataSync()[ans] * 100) + '%');
            console.log('Debug Mode On');
            this.debugDrawCorrespondence(0, 0, outputs, ans, image)
                .then((corr_image) => resolve({
                    type: 'custom',
                    site: this.label[outputs[2].dataSync()[ans] - 1],
                    confidence: Math.round(outputs[1].dataSync()[ans] * 100),
                    corr_img: corr_image,
                    display_confidence: ': ' + Math.round(outputs[1].dataSync()[ans] * 100) + '%'
                }));

        });
    }

    async debugPredict(screenshot, page_url) {
        return new Promise((resolve, reject) => {
            console.log('I am inside tfJS');
            var t0, t1, t11, t01, tf_img, image;
            this.debugLoadImage(screenshot)
                .then(img => {
                    t0 = performance.now();
                    image = img;
                    return tf.browser.fromPixels(image);
                })
                .then(tf_image => {
                    tf.print(tf_image);
                    tf_img = tf_image.expandDims(0);
                    console.log(tf_img.shape);
                    t1 = performance.now();
                    console.log("Image to tensor took " + (t1 - t0) + " milliseconds.");
                    return tf_img;
                })
                .then((tf_img) => {
                    let backend = 'webgl';
                    console.log('tf backend ' + backend + ' configured');
                    tf.setBackend(backend);
                    t01 = performance.now();
                    return this.model.executeAsync({
                        'image_tensor': tf_img
                    }, ['detection_boxes', 'detection_scores', 'detection_classes']);
                })
                .then(outputs => {
                    t11 = performance.now();
                    console.log("Prediction took " + (t11 - t01) + " milliseconds.");
                    for (let i = 0; i < outputs.length; i++) {
                        console.log(outputs[i].dataSync());
                    }
                    return outputs;
                })
                .then((outputs) => this.debugDisplayPrediction(outputs, image))
                .then((pred_result) => {
                    this.count = this.count + 1;
                    // report result to System Logs@track-result.js
                    // if (pred_result != null) {
                    //     report.trackResult({site: pred_result.site, confidence: pred_result.confidence, time_taken: (Math.round((t11 - t01)/1000))}, pred_result.corr_image, report.getLocation(page_url));
                    // }
                    // else {
                    //     report.trackResult({site: 'NaN', confidence: 0, time_taken: Math.round((t11 - t01)/1000)}, screenshot, report.getLocation(page_url));
                    // }
                    console.log("Result: " + pred_result);
                    resolve(pred_result)
                })
                .catch(() => {
                    console.log("No Label found for this Site! :(");
                    resolve(false);
                });
        });
    }

    debugLoadImage(imageUrl) {
        // * Loads the image by making a HTML Image Element from the obtaied B64
        return new Promise((resolve, reject) => {
            let img = new Image();
            img.crossOrigin = '';
            img.onload = () => {
                resolve(img);
            };
            img.onerror = () => {
                reject("Error: Unable to load image");
            };
            img.src = imageUrl;
        });
    }

}
mobilenetv2Classification.dependencies = [
    "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.6.0/dist/tf.min.js",
]
mobilenetv2Classification.model = ROOT_DIR + "/mobilenetv2Classification/model/model.json"
