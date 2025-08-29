# -*- coding: utf-8 -*-
"""
geo_database.py

Türkiye adres çözümleme için temel coğrafi veritabanı:
- 81 il ve sık geçen bazı ilçeler (liste kısmi ama geniş)
- Şehir kısaltmaları / varyasyonları
- Posta kodu ilk 2 hane -> il eşlemesi (PTT sistemi)

Kullanım:
from geo_database import GeoDatabase
geo = GeoDatabase()
geo.provinces               # dict[str, list[str]]
geo.city_variations         # dict[str, str]
geo.postal_prefixes         # dict[str, str]

Ayrıca yardımcı metotlar:
- find_province(text)
- find_district(text)
- province_from_postal(text)
"""
from __future__ import annotations
import re
from typing import Dict, List

class GeoDatabase:
    def __init__(self):
        # 81 il -> örnek/ana ilçeler (tam liste değil; sık geçenler)
        self.provinces: Dict[str, List[str]] = {
            'adana': ['seyhan','yuregir','cukurova','karaisali','karatas'],
            'adiyaman': ['merkez','besni','celikhan','gerger','golbasi','kahta','samsat','sincik','tut'],
            'afyonkarahisar': ['merkez','sandikli','dinar','bolvadin','cay','dazkiri','emirdag'],
            'agri': ['merkez','dogubayazit','patnos','tutak','diyadin','eleskirt','hamur','taslicay'],
            'amasya': ['merkez','merzifon','suluova','tasova','gokdere','hamamozu'],
            'ankara': ['cankaya','kecioren','yenimahalle','mamak','altindag','sincan','etimesgut','golbasi','polatli','pursaklar','akyurt','ayas','bala','beypazari'],
            'antalya': ['muratpasa','kepez','konyaalti','alanya','manavgat','serik','kas','kemer','elmali','finike','gazipasa','demre','kumluca','akseki'],
            'artvin': ['merkez','hopa','borcka','arhavi','yusufeli','savsat','ardanuc','murgul'],
            'aydin': ['efeler','nazilli','soke','kusadasi','didim','cine','bozdogan','germencik'],
            'balikesir': ['karesi','altieylul','bandirma','edremit','ayvalik','burhaniye','erdek'],
            'bartin': ['merkez','amasra','kurucasile','ulus'],
            'batman': ['merkez','kozluk','besiri','gercus','hasankeyf','sason'],
            'bayburt': ['merkez','aydintepe','demirozu'],
            'bilecik': ['merkez','bozuyuk','sogut','osmaneli','golpazari','inhisar','pazaryeri','yenipazar'],
            'bingol': ['merkez','genc','karliova','solhan','adakli','kigi','yayladere','yedisu'],
            'bitlis': ['merkez','tatvan','guroymak','hizan','mutki','adilcevaz','ahlat'],
            'bolu': ['merkez','goeynuek','mudurnu','mengen','gerede','kibriscik','seben'],
            'burdur': ['merkez','bucak','golhisar','yesilova','karamanli','aglasun','altinyayla'],
            'bursa': ['osmangazi','nilufer','yildirim','mudanya','gemlik','inegol','orhaneli','buyukorhan','harmancik','iznik','karacabey','keles','kestel','mustafakemalpasa'],
            'canakkale': ['merkez','gelibolu','biga','can','ayvacik','bayramic','bozcaada'],
            'cankiri': ['merkez','cerkes','ilgaz','kursunlu','orta','atkaracalar','bayramoren'],
            'corum': ['merkez','osmancik','iskilip','kargi','dodurga','alaca','bayat','bogazkale'],
            'denizli': ['pamukkale','merkezefendi','honaz','tavas','cal','acipayam','buldan','cameli'],
            'diyarbakir': ['baglar','kayapinar','sur','yenisehir','bismil','cermik','cinar','dicle'],
            'edirne': ['merkez','kesan','uzunkopru','ipsala','havsa','enez','lalapasa','meric','suloglu'],
            'elazig': ['merkez','karakocan','keban','palu','sivrice','agin','alacakaya','aricak','baskil'],
            'erzincan': ['merkez','uzumlu','refahiye','tercan','kemah','cayirli','ilic','kemaliye','otlukbeli'],
            'erzurum': ['yakutiye','palandoken','aziziye','hinis','pasinler','askale','cat','horasan'],
            'eskisehir': ['tepebasi','odunpazari','sivrihisar','cifteler','alpu','beylikova'],
            'gaziantep': ['sahinbey','sehitkamil','nizip','islahiye','nurdagi','araban','karkamis'],
            'giresun': ['merkez','bulancak','espiye','gorele','tirebolu','alucra','camoluk','canakci'],
            'gumushane': ['merkez','kelkit','siran','torul','kose','kurtun'],
            'hakkari': ['merkez','yuksekova','semdinli','cukurka'],
            'hatay': ['antakya','defne','arsuz','dortyol','iskenderun','kirikhan','payas','reyhanli','samandag'],
            'igdir': ['merkez','tuzluca','karakoyunlu','aralik'],
            'isparta': ['merkez','yalvac','keciborlu','sarkikaraagac','atabey','egirdir','gelendost'],
            'istanbul': ['fatih','beyoglu','uskudar','kadikoy','besiktas','sisli','bakirkoy','zeytinburnu','esenler','gaziosmanpasa','kagithane','sariyer','maltepe','pendik','umraniye','beykoz','beylikduzu','esenyurt','avcilar','kucukcekmece','buyukcekmece','bahcelievler','bagcilar','gungoren','sultangazi','arnavutkoy','basaksehir','bayrampasa','eyupsultan','kagithane','kartal','silivri','sultanbeyli'],
            'izmir': ['konak','bornova','cigli','karsiyaka','bayrakli','gaziemir','balcova','narlidere','buca','menderes','torbali','menemen','foca','urla','cesme','odemis','tire','bergama','dikili','karaburun','kinik','kiraz','selcuk','aliaga'],
            'kahramanmaras': ['onikisubat','dulkadiroglu','pazarcik','elbistan','afsin','andirin'],
            'karabuk': ['merkez','safranbolu','yenice','eskipazar','ovacik','eflani'],
            'karaman': ['merkez','ermenek','kazimkarabekir','basyayla','ayranci','sariveliler'],
            'kars': ['merkez','kagizman','ardahan','arpacay','selim','akyaka','digor','susuz'],
            'kastamonu': ['merkez','taskopru','sinop','boyabat','cankiri','abana','agli','arac'],
            'kayseri': ['melikgazi','kocasinan','talas','develi','yahyalı','bunyan','felahiye','hacilar'],
            'kirikkale': ['merkez','yahsihan','keskin','sulakyurt','bahsili','baliseyh','celebi','delice'],
            'kirklareli': ['merkez','babaeski','luleburgaz','pinarhisar','demirkoy','kofcaz','pehlivankoy','vize'],
            'kirsehir': ['merkez','kaman','mucur','cicekdag','akcakent','akpinar','boztepe'],
            'kilis': ['merkez','musabeyli','polateli','elbeyli'],
            'kocaeli': ['izmit','gebze','darica','korfez','golcuk','kandira','basiskele','cayirova','derince','dilovasi','kartepe'],
            'konya': ['selcuklu','meram','karatay','eregli','aksehir','beysehir','cumra','ilgin','kulu'],
            'kutahya': ['merkez','tavsanli','gediz','simav','emet','altintas','aslanapa','cavdarhisar'],
            'malatya': ['yesilyurt','battalgazi','akcadag','darende','dogansehir','hekimhan','kuluncak','puturge'],
            'manisa': ['yunusemre','sehzadeler','turgutlu','akhisar','salihli','alasehir','demirci','gordes'],
            'mardin': ['artuklu','midyat','kiziltepe','nusaybin','omerli','dargecit','derik','mazidagi'],
            'mersin': ['yenisehir','mezitli','toroslar','akdeniz','tarsus','erdemli','anamur','aydincik'],
            'mugla': ['mentese','bodrum','marmaris','fethiye','milas','ortaca','cine','datca','koycegiz'],
            'mus': ['merkez','bulanik','malazgirt','varto','haskoy','korkut'],
            'nevsehir': ['merkez','urgup','avanos','goreme','derinkuyu','acigol','cat','hacibektas'],
            'nigde': ['merkez','bor','ulukisla','camardi','altunhisar','ciftlik'],
            'ordu': ['altinordu','unye','fatsa','persembe','caybasi','akkus','yabasit','camas'],
            'osmaniye': ['merkez','kadirli','duzici','bahce','hasanbeyli','sumbas','toprakkale'],
            'rize': ['merkez','cayeli','ardesen','pazar','findikli','guneysu','hemsin','ikizdere'],
            'sakarya': ['adapazari','serdivan','akyazi','karasu','hendek','arifiye','erenler','ferizli','geyve'],
            'samsun': ['ilkadim','canik','atakum','bafra','carsamba','terme','alacam','asarci'],
            'sanliurfa': ['eyyubiye','haliliye','karakopru','viransehir','suruc','akcakale','birecik'],
            'siirt': ['merkez','kurtalan','sirvan','baykan','pervari','aydinlar','eruh'],
            'sinop': ['merkez','boyabat','ayancik','turkeli','erfelek','duragan','gerze'],
            'sirnak': ['merkez','cizre','silopi','idil','guclukonak','beytussebap','uludere'],
            'sivas': ['merkez','sarkisla','yildizeli','susehri','divrigi','gemerek','gurun','hafik'],
            'tekirdag': ['suleymanpasa','corlu','cerkezkoy','hayrabolu','kapakli','malkara','marmaraereglisi','muratli','saray','sarkoy'],
            'tokat': ['merkez','turhal','erbaa','niksar','resadiye','almus','artova','basciftlik'],
            'trabzon': ['ortahisar','akcaabat','vakfikebir','of','arakli','arsin','besikduzu','caykara'],
            'tunceli': ['merkez','cemisgezek','hozat','mazgirt','ovacik','pertek','pulumur'],
            'usak': ['merkez','banaz','esme','karahalli','sivasli','ulubey'],
            'van': ['ipekyolu','tusba','edremit','gevas','muradiye','bahcesaray','baskale','caldiran'],
            'yalova': ['merkez','ciftlikkoy','altinova','armutlu','cinarcik','termal'],
            'yozgat': ['merkez','sorgun','bogazliyan','yerkoy','sefaatli','akdagmadeni','aydincik'],
            'zonguldak': ['merkez','eregli','caycuma','devrek','gokcebey','alapli','kilimli']
        }

        self.city_variations: Dict[str,str] = {
            'ist':'istanbul','izm':'izmir','ank':'ankara','bur':'bursa','ada':'adana','ant':'antalya','mer':'mersin',
            'gaz':'gaziantep','kon':'konya','kay':'kayseri','den':'denizli','man':'manisa','tra':'trabzon','sam':'samsun',
            'esk':'eskisehir','mal':'malatya',
            # yaygın yazım bozuklukları
            'uskuar':'uskudar','kadkoy':'kadikoy','galtsaray':'galatasaray','taksm':'taksim','umranye':'umraniye'
        }

        self.postal_prefixes: Dict[str,str] = {
            '01':'adana','02':'adiyaman','03':'afyonkarahisar','04':'agri','05':'amasya','06':'ankara','07':'antalya','08':'artvin',
            '09':'aydin','10':'balikesir','11':'bilecik','12':'bingol','13':'bitlis','14':'bolu','15':'burdur','16':'bursa',
            '17':'canakkale','18':'cankiri','19':'corum','20':'denizli','21':'diyarbakir','22':'edirne','23':'elazig','24':'erzincan',
            '25':'erzurum','26':'eskisehir','27':'gaziantep','28':'giresun','29':'gumushane','30':'hakkari','31':'hatay','32':'isparta',
            '33':'mersin','34':'istanbul','35':'izmir','36':'kars','37':'kastamonu','38':'kayseri','39':'kirklareli','40':'kirsehir',
            '41':'kocaeli','42':'konya','43':'kutahya','44':'malatya','45':'manisa','47':'mardin','48':'mugla','49':'mus',
            '50':'nevsehir','51':'nigde','52':'ordu','53':'rize','54':'sakarya','55':'samsun','56':'siirt','57':'sinop',
            '58':'sivas','59':'tekirdag','60':'tokat','61':'trabzon','62':'tunceli','63':'sanliurfa','64':'usak','65':'van',
            '66':'yozgat','67':'zonguldak','68':'aksaray','69':'bayburt','70':'karaman','71':'kirikkale','72':'batman','73':'sirnak',
            '74':'bartin','75':'ardahan','76':'igdir','77':'yalova','78':'karabuk','79':'kilis','80':'osmaniye','81':'duzce'
        }

        self._postal_re = re.compile(r"\b(\d{5})\b")

    # ---------- helpers ----------
    def find_province(self, text: str) -> str:
        t = (text or '').lower()
        for p in self.provinces.keys():
            if p in t:
                return p
        # try variations
        for k,v in self.city_variations.items():
            if f" {k} " in f" {t} ":
                return v
        # try postal
        pv = self.province_from_postal(t)
        return pv or ''

    def find_district(self, text: str) -> str:
        t = (text or '').lower()
        for p, ds in self.provinces.items():
            for d in ds:
                if d in t:
                    return d
        return ''

    def province_from_postal(self, text: str) -> str:
        m = self._postal_re.search(text or '')
        if not m:
            return ''
        pref = m.group(1)[:2]
        return self.postal_prefixes.get(pref,'')
