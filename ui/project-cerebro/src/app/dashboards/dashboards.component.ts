import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';

interface DashboardURLs {
  grafanaURL: string;
  jupyterURL: string;
  tensorboardURL: string;
}

@Component({
  selector: 'app-dashboards',
  templateUrl: './dashboards.component.html',
  styleUrls: ['./dashboards.component.css']
})

export class DashboardsComponent implements OnInit{
  constructor(private httpClient: HttpClient){}
  urls = <DashboardURLs> {};

  ngOnInit() {
    const baseURL = environment.backendURL;
    
    this.httpClient.get(baseURL + '/get-urls').subscribe((res: any)=>{
      this.urls.grafanaURL = res.message["grafanaURL"];
      this.urls.jupyterURL = res.message["jupyterURL"];
      this.urls.tensorboardURL = res.message["tensorboardURL"];
      console.log(this.urls);
    });
  }

  goToLink(url: string){
    window.open(url, "_blank");
  };
}