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
      
    }
}

$(document).ready(function(e) {
    var i = new Index();
    i.hookSlidingBars();
});
