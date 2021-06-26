/*
Main page
=========
*/
'use strict';

import _ from 'lodash';
import axios from 'axios';
import dateformat from 'dateformat';
import 'bootstrap';
import 'bootstrap/js/dist/tooltip';
import 'selectize';
import randomColor from 'randomcolor';
import 'datatables.net';
import 'datatables.net-dt';
import gradient from 'gradient-color';
import 'bootstrap4-toggle';
import 'jquery-ui-bundle';
import Plotly from 'plotly.js-dist';
import Slider from 'bootstrap-slider';

class Index {
    constructor(env) {

    }
    hookSlidingBars() {
      var bars = [
        "time_on_site", "total_view_products",
        "unique_add_to_cart", "mean_product_price", "total_searches",
        "total_add_to_cart", "hour_of_day"
      ];

      bars.forEach(function (item, index) {
        console.log(item, index);
        var mySlider = new Slider("#"+item, {
          formatter: function(value) {
            return 'Current value: ' + value;
          }
        });
      });
    }

    user_conversion_predict(){
      var url = '/user_conversion_predict' +
        '?time_on_site=' + $("#time_on_site").val() +
        '&total_view_products=' + $("#total_view_products").val() +
        '&unique_add_to_cart=' + $("#unique_add_to_cart").val() +
        '&mean_product_price=' + $("#mean_product_price").val() +
        '&total_searches=' + $("#total_searches").val() +
        '&total_add_to_cart=' + $("#total_add_to_cart").val() +
        '&hour_of_day=' + $("#hour_of_day").val();
      axios.get(url).then(function(res) {
        console.log(res);
        $("#convert").html(res.data["convert"]);
        $("#nonconvert").html(res.data["nonconvert"]);
      });
    }
}

$(document).ready(function(e) {
    var i = new Index();
    i.hookSlidingBars();
    $("#predict_button").click(function(){
      i.user_conversion_predict();
    });
});
