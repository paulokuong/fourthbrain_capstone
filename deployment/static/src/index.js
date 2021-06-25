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

class Index {
    constructor(env) {

    }
    hookSlidingBars() {
      var bars = [
        "time_on_site", "total_view_products",
        "unique_add_to_cart", "mean_product_price", "total_searches",
        "total_add_to_cart", "hour_of_day"
      ];
      for(var bar in bars){
        console.log(bar);
        $('#'+bar).slider({
          formatter: function(value) {
            return 'Current value: ' + value;
          }
        });
      }
    }
}

$(document).ready(function(e) {
    var i = new Index();
    i.hookSlidingBars();
    $('[data-toggle="tooltip"]').tooltip();
    $('#carousel').carousel({
        interval: 2000
    })
});
