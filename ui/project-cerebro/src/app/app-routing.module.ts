import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { CodeComponent } from './code/code.component';
import { DashboardsComponent } from './dashboards/dashboards.component';
import { LandingComponent } from './landing/landing.component';
import { ParamsComponent } from './params/params.component';
 
const routes: Routes = [
  { path: 'params', component: ParamsComponent },
  { path: 'code', component: CodeComponent },
  { path: 'landing', component: LandingComponent},
  { path: 'dashboards', component: DashboardsComponent},
  { path: '', redirectTo: '/landing', pathMatch: 'full'}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
