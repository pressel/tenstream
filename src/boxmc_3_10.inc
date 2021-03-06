!-------------------------------------------------------------------------
! This file is part of the tenstream solver.
!
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.
!
! Copyright (C) 2010-2015  Fabian Jakub, <fabian@jakub.com>
!-------------------------------------------------------------------------


subroutine update_dir_stream_3_10(bmc,p,T)
        class(t_boxmc_3_10) :: bmc
        type(photon),intent(in) :: p
        real(ireal_dp),intent(inout) :: T(:)

    select case (p%side)
    case(1:2)
      if(p%loc(1).le.p%dx.and.p%loc(2).le.p%dy) then
        T(1) = T(1)+p%weight
        return
      else
        print *,'Couldnt find a stream on to which I can put the photon weight on?!'
        call print_photon(p)
      endif

    case(3:4)

      if(p%loc(3).le.p%dz) then
        T(2) = T(2)+p%weight
        return
      else
        print *,'Couldnt find a stream on to which I can put the photon weight on?!'
        call print_photon(p)
      endif
    case(5:6)

      if(p%loc(3).le.p%dz ) then
        T(3) = T(3)+p%weight
        return
      else
        print *,'Couldnt find a stream on to which I can put the photon weight on?!'
        call print_photon(p)
      endif
    case default
      print *,'Dont know what to do with this p%side'
      call print_photon(p)
    end select
        select type(bmc)
        end select

end subroutine
subroutine update_diff_stream_3_10(bmc,p,S)
        class(t_boxmc_3_10) :: bmc
        type(photon),intent(in) :: p
        real(ireal_dp),intent(inout) :: S(:)

!         _______1_______
!        |           10  |
!      5 |            8  | 6
!        |               |
!        |   9           |
!        |   7           |
!      3 |_______________| 4
!                2

        if(p%side.eq.1) then
                S(1) = S(1)+p%weight
                return
        else if(p%side.eq.2) then
                S(2) = S(2)+p%weight
                return

        else if(p%side.eq.3 .and. p%dir(3).le.zero ) then
                S(3) = S(3)+p%weight
                return
        else if(p%side.eq.3 .and. p%dir(3).gt.zero ) then
                S(5) = S(5)+p%weight
                return

        else if(p%side.eq.4 .and. p%dir(3).le.zero ) then
                S(4) = S(4)+p%weight
                return
        else if(p%side.eq.4 .and. p%dir(3).gt.zero ) then
                S(6) = S(6)+p%weight
                return

        else if(p%side.eq.5 .and. p%dir(3).le.zero ) then
                S(7) = S(7)+p%weight
                return
        else if(p%side.eq.6 .and. p%dir(3).le.zero ) then
                S(8) = S(8)+p%weight
                return
        else if(p%side.eq.5 .and. p%dir(3).gt.zero ) then
                S(9) = S(9)+p%weight
                return
        else if(p%side.eq.6 .and. p%dir(3).gt.zero ) then
                S(10) =S(10)+p%weight
                return
        else
                print *,'Couldnt find a stream on to which I can put the photon weight on?!'
                call print_photon(p)
                call exit
        endif
        select type(bmc)
        end select

end subroutine

subroutine init_dir_photon_3_10(bmc,p,src,direct,initial_dir,dx,dy,dz)
        class(t_boxmc_3_10) :: bmc
        type(photon),intent(inout) :: p
        real(ireal_dp),intent(in) :: dx,dy,dz,initial_dir(3)
        integer(iintegers),intent(in) :: src
        logical,intent(in) :: direct

        p%alive = .False.

        select case(src)
        case (1)
                p%loc = (/L(dx)     , L(dy)     ,    dz  /)
        case (2)
                p%loc = (/  zero    , L(dy)     , L(dz)  /)
        case (3)
                p%loc = (/L(dx)     ,   zero    , L(dz)  /)
        case default
                print *,'Dont know what to do with source spec:',src
                call exit
        end select

        p%weight=one
        p%dx   = dx
        p%dy   = dy
        p%dz   = dz
        p%alive = .True.
        p%direct= direct
        p%side = int(nil)
        p%src  = src
        p%scattercnt  = 0
        p%dir = initial_dir
        select type(bmc)
        end select


end subroutine

subroutine init_diff_photon_3_10(bmc,p,src,dx,dy,dz)
        class(t_boxmc_3_10) :: bmc
        type(photon),intent(inout) :: p
        real(ireal_dp),intent(in) :: dx,dy,dz
        integer(iintegers),intent(in) :: src

        real(ireal_dp) :: phi,mu

        if(src.eq.1) then
                p%loc = (/L(dx), L(dy),    dz  /)
                mu = interv_R( deg2mu(90._ireal_dp),deg2mu(180._ireal_dp) ) ; phi= deg2rad( R()*360._ireal_dp )
        else if(src.eq.2) then
                p%loc = (/L(dx), L(dy),    zero /)
                mu = interv_R( deg2mu(0._ireal_dp),deg2mu(90._ireal_dp) ) ; phi= deg2rad( R()*360._ireal_dp )
        else if(src.eq.3) then
                p%loc = (/  zero   , L(dy), L(dz)/)
                mu = interv_R( deg2mu(90._ireal_dp),deg2mu(180._ireal_dp) ) ; phi= deg2rad( interv_R(-90._ireal_dp,90._ireal_dp) )
        else if(src.eq.4) then
                p%loc = (/ dx   , L(dy), L(dz)/)
                mu = interv_R( deg2mu(90._ireal_dp),deg2mu(180._ireal_dp) ) ; phi= deg2rad( interv_R(90._ireal_dp,270._ireal_dp) )
        else if(src.eq.5) then
                p%loc = (/  zero   , L(dy), L(dz)/)
                mu = interv_R( deg2mu(0._ireal_dp),deg2mu(90._ireal_dp) ) ; phi= deg2rad( interv_R(-90._ireal_dp,90._ireal_dp) )
        else if(src.eq.6) then
                p%loc = (/ dx   , L(dy), L(dz)/)
                mu = interv_R( deg2mu(0._ireal_dp),deg2mu(90._ireal_dp) ) ; phi= deg2rad( interv_R(90._ireal_dp,270._ireal_dp) )
        else if(src.eq.7) then
                p%loc = (/L(dx),   zero   , L(dz)/)
                mu = interv_R( deg2mu(90._ireal_dp),deg2mu(180._ireal_dp) ) ; phi= deg2rad( interv_R(0._ireal_dp,180._ireal_dp) )
        else if(src.eq.8) then
                p%loc = (/L(dx),  dy   , L(dz)/)
                mu = interv_R( deg2mu(90._ireal_dp),deg2mu(180._ireal_dp) ) ; phi= deg2rad( interv_R(180._ireal_dp,360._ireal_dp) )
        else if(src.eq.9) then
                p%loc = (/L(dx),   zero   , L(dz)/)
                mu = interv_R( deg2mu(0._ireal_dp),deg2mu(90._ireal_dp) ) ; phi= deg2rad( interv_R(0._ireal_dp,180._ireal_dp) )
        else if(src.eq.10) then
                p%loc = (/L(dx),  dy   , L(dz)/)
                mu = interv_R( deg2mu(0._ireal_dp),deg2mu(90._ireal_dp) ) ; phi= deg2rad( interv_R(180._ireal_dp,360._ireal_dp) )
        else
                print *,'Dont know what to do with source spec:',src
                call exit
        endif

        p%dir = (/cos(phi)*sin(acos(mu)) , sin(phi)*sin(acos(mu)) , mu  /)
        p%weight=one
        p%dx   = dx
        p%dy   = dy
        p%dz   = dz
        p%alive = .True.
        p%direct= .False.
        p%side = int(nil)
        p%src  = src
        p%scattercnt  = 0
        if(.not. approx( norm(p%dir) ,one ) ) stop 'init_diff_photon_3_10 :: direction was not normed?!'

        select type(bmc)
        end select

end subroutine

subroutine intersect_distance_3_10(bmc,p,max_dist)
        class(t_boxmc_3_10) :: bmc
        type(photon),intent(inout) :: p
        real(ireal_dp),intent(out) :: max_dist

        real(ireal_dp) :: x,y,z
        integer(iintegers) :: i,sides(3)

        real(ireal_dp) :: dist(3)

        !crossing with bottom and top plane:
          if(p%dir(3).ge.zero) then
            max_dist = hit_plane(p%loc, p%dir,[zero,zero,p%dz ],[zero,zero,one])
            p%side=1
            x = p%loc(1)+p%dir(1)*max_dist
            y = p%loc(2)+p%dir(2)*max_dist
            if( ( x.ge.zero .and. x.le.p%dx) .and. ( y.ge.zero .and. y.le.p%dy) ) return
            dist(1) = max_dist; sides(1) = 1
          endif
          if(p%dir(3).le.zero) then
            max_dist = hit_plane(p%loc, p%dir,[zero,zero,zero ],[zero,zero,one])
            p%side=2
            x = p%loc(1)+p%dir(1)*max_dist
            y = p%loc(2)+p%dir(2)*max_dist
            if( ( x.ge.zero .and. x.le.p%dx) .and. ( y.ge.zero .and. y.le.p%dy) ) return
            dist(1) = max_dist; sides(1) = 2
          endif

          !crossing with left and right plane:
          if(p%dir(1).le.zero) then
            max_dist = hit_plane(p%loc, p%dir,[ zero ,zero,zero],[one,zero,zero])
            p%side=3
            y = p%loc(2)+p%dir(2)*max_dist
            z = p%loc(3)+p%dir(3)*max_dist
            if( ( y.ge.zero .and. y.le.p%dy) .and. ( z.ge.zero .and. z.le.p%dz) ) return
            dist(2) = max_dist; sides(2) = 3
          endif
          if(p%dir(1).ge.zero) then
            max_dist = hit_plane(p%loc, p%dir,[ p%dx ,zero,zero],[one,zero,zero])
            p%side=4
            y = p%loc(2)+p%dir(2)*max_dist
            z = p%loc(3)+p%dir(3)*max_dist
            if( ( y.ge.zero .and. y.le.p%dy) .and. ( z.ge.zero .and. z.le.p%dz) ) return
            dist(2) = max_dist; sides(2) = 4
          endif

          !crossing with back and forward plane:
          if(p%dir(2).le.zero) then
            max_dist = hit_plane(p%loc, p%dir,[zero, zero ,zero],[zero,one,zero])
            p%side=5
            x = p%loc(1)+p%dir(1)*max_dist
            z = p%loc(3)+p%dir(3)*max_dist
            if( ( x.ge.zero .and. x.le.p%dx) .and. ( z.ge.zero .and. z.le.p%dz) ) return
            dist(3) = max_dist; sides(3) = 5
          endif
          if(p%dir(2).ge.zero) then
            max_dist = hit_plane(p%loc, p%dir,[zero, p%dy ,zero],[zero,one,zero])
            p%side=6
            x = p%loc(1)+p%dir(1)*max_dist
            z = p%loc(3)+p%dir(3)*max_dist
            if( ( x.ge.zero .and. x.le.p%dx) .and. ( z.ge.zero .and. z.le.p%dz) ) return
            dist(3) = max_dist; sides(3) = 6
          endif

          !Ohhh there was a problem.. maybe with numerics, seems that it may happen that we dont find a solution if norm of p%dir is not equal to one....
          max_dist=huge(dist)
          do i=1,3
            if(.not. approx(p%dir(i),zero) ) then
              if( dist(i).le.max_dist ) then
                p%side = sides(i)
                max_dist = dist(i)
              endif
            endif
          enddo

          !print *,'should actually not be here at the end of crossings in intersect distance! - however, please check if distance makes sense?:',max_dist
          !call print_photon(p)

        select type(bmc)
        end select

end subroutine
  ! ******************* BOXMC 3 10 ************************

